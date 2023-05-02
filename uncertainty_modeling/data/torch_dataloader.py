from abc import ABC
from typing import Tuple, Dict, Any

import hydra
from omegaconf import DictConfig
import numpy as np

from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

import torch
import random

import uncertainty_modeling.data.cityscapes_labels as cs_labels

# set number of Threads to 0 for opencv and albumentations
cv2.setNumThreads(0)
# import logger
# log = get_logger(__name__)


class StochasticLabelSwitches(A.BasicTransform):
    def __init__(self, always_apply=False, p=0.5):
        super(StochasticLabelSwitches, self).__init__(always_apply, p)
        self._name2id = cs_labels.name2trainId
        self._label_switches = {
            "sidewalk": 1.0 / 3.0,
            "person": 1.0 / 3.0,
            "car": 1.0 / 3.0,
            "vegetation": 1.0 / 3.0,
            "road": 1.0 / 3.0,
        }

    def apply(self, img, **params):
        return img

    def apply_to_mask(self, mask, **params):
        for c, p in self._label_switches.items():
            init_id = self._name2id[c]
            final_id = self._name2id[c + "_2"]
            switch_instances = np.random.binomial(1, p, 1)

            if switch_instances[0]:
                mask[mask == init_id] = final_id
        return mask

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    def get_transform_init_args_names(self):
        return ()

    @property
    def targets(self):
        return {"mask": self.apply_to_mask}


def seed_worker(worker_id):
    """
    from: https://github.com/MIC-DKFZ/image_classification/blob/master/base_model.py
        https://pytorch.org/docs/stable/notes/randomness.html#dataloader
        to fix https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/
        ensures different random numbers each batch with each worker every epoch while keeping reproducibility
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_max_steps(
    size_dataset,
    batch_size,
    num_devices,
    accumulate_grad_batches,
    num_epochs,
    drop_last=True,
) -> int:
    """
    Computing the number of  steps, needed for polynomial lr scheduler
    Considering the number of gpus and if accumulate_grad_batches is used

    Returns
    -------
    int:
        total number of steps
    int:
        number of steps per epoch
    """
    # How many steps per epoch in total
    if drop_last:
        steps_per_epoch = size_dataset // batch_size  # round off if drop_last=False
    else:
        steps_per_epoch = np.ceil(
            size_dataset / batch_size
        )  # round up if drop_last=False

    # For ddp and multiple gpus the effective batch sizes doubles
    steps_per_gpu = int(np.ceil(steps_per_epoch / num_devices))
    # Include accumulate_grad_batches
    steps_per_epoch = int(np.ceil(steps_per_gpu / accumulate_grad_batches))
    max_steps = num_epochs * steps_per_epoch

    return max_steps, steps_per_epoch


def get_augmentations() -> list:
    """
    Build an Albumentations augmentation pipeline from the input config

    Parameters
    ----------
    augmentations : DictConfig
        config of the Augmentation

    Returns
    -------
    list :
        list of Albumentations transforms
    """
    transforms = [
        A.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], always_apply=True
        ),
        StochasticLabelSwitches(always_apply=True, p=1.0),
        ToTensorV2(always_apply=True, p=1.0, transpose_mask=False),
    ]
    return A.Compose(transforms)


class BaseDataModule(LightningDataModule):
    def __init__(
        self,
        data_input_dir: str,
        dataset,
        batch_size: int,
        val_batch_size: int,
        num_workers: int,
        **kwargs
    ) -> None:
        """
        __init__ the LightningModule
        save parameters

        Parameters
        ----------
        dataset : DictConfig
            config of the dataset, is called by hydra.src.instantiate(dataset,split=.., transforms=..)
        batch_size : int
            batch size for train dataloader
        val_batch_size : int
            batch size for val and test dataloader
        num_workers : int
            number of workers for all dataloaders
        augmentations : DictConfig
            config containing the augmentations for Train, Test and Validation
        """
        super().__init__()

        # parameters for dataloader
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.augmentations = get_augmentations()
        self.data_input_dir = data_input_dir
        # dataset which is defined in the config
        self.dataset = dataset

    def setup(self, stage: str = None) -> None:
        """
        Setting up the Datasets by initializing the augmentation and the dataloader

        Parameters
        ----------
        stage: str
            current stage which is given by Pytorch Lightning
        """
        self.DS_train = hydra.utils.instantiate(
            self.dataset,
            data_input_dir=self.data_input_dir,
            split="train",
            transforms=get_augmentations(),
        )
        self.DS_val = hydra.utils.instantiate(
            self.dataset,
            data_input_dir=self.data_input_dir,
            split="val",
            transforms=get_augmentations(),
        )
        self.DS_test = hydra.utils.instantiate(
            self.dataset,
            data_input_dir=self.data_input_dir,
            split="test",
            transforms=get_augmentations(),
        )

    def max_steps(self) -> int:
        """
        Computing and Logging the number of training steps, needed for polynomial lr scheduler
        Considering the number of gpus and if accumulate_grad_batches is used

        Returns
        -------
        int:
            number of training steps
        """
        # computing the maximal number of steps for training
        max_steps, max_steps_epoch = get_max_steps(
            size_dataset=len(self.DS_train),
            batch_size=self.batch_size,
            num_devices=self.trainer.num_devices,
            accumulate_grad_batches=self.trainer.accumulate_grad_batches,
            num_epochs=self.trainer.max_epochs,
            drop_last=True,
        )
        # base_size = len(self.DS_train)
        # steps_per_epoch = base_size // self.batch_size
        # steps_per_gpu = int(np.ceil(steps_per_epoch / self.trainers.num_devices))
        # acc_steps_per_gpu = int(np.ceil(steps_per_gpu / self.trainers.accumulate_grad_batches))
        # max_steps = self.trainers.max_epochs * acc_steps_per_gpu

        print(
            "Number of Training steps: {}  ({} steps per epoch)".format(
                max_steps, max_steps_epoch
            )
        )

        max_steps_val, max_steps_epoch_val = get_max_steps(
            size_dataset=len(self.DS_val),
            batch_size=self.val_batch_size,
            num_devices=self.trainer.num_devices,
            accumulate_grad_batches=1,
            num_epochs=self.trainer.max_epochs,
            drop_last=False,
        )

        print(
            "Number of Validation steps: {}  ({} steps per epoch)".format(
                max_steps_val, max_steps_epoch_val
            )
        )
        return max_steps

    def train_dataloader(self) -> DataLoader:
        """
        Returns
        -------
        DataLoader :
            train dataloader
        """
        return DataLoader(
            self.DS_train,
            shuffle=True,
            pin_memory=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            persistent_workers=True if self.num_workers > 0 else False,
            worker_init_fn=seed_worker,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Returns
        -------
        DataLoader :
            validation dataloader
        """
        return DataLoader(
            self.DS_val,
            pin_memory=True,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Returns
        -------
        DataLoader :
            test dataloader
        """
        return DataLoader(
            self.DS_test,
            pin_memory=True,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )
