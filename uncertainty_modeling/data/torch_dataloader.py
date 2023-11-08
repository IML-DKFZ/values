"""
------------------------------------------------------------------------------
Code slightly adapted and mainly from:
https://github.com/MIC-DKFZ/semantic_segmentation/blob/public/datasets/DataModules.py
------------------------------------------------------------------------------
"""

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

import uncertainty_modeling.augmentations as custom_augmentations

# set number of Threads to 0 for opencv and albumentations
cv2.setNumThreads(0)


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


def get_augmentations_from_config(augmentations: DictConfig) -> list:
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
    # otherwise recursively build the transformations
    trans = []
    for augmentation in augmentations:
        transforms = list(augmentation.keys())

        for transform in transforms:
            parameters = getattr(augmentation, transform)
            if parameters is None:
                parameters = {}

            if hasattr(A, transform):
                if "transforms" in list(parameters.keys()):
                    # "transforms" indicates a transformation which takes a list of other transformations
                    # as input ,e.g. A.Compose -> recursively build these transforms
                    transforms = get_augmentations_from_config(parameters.transforms)
                    del parameters["transforms"]
                    func = getattr(A, transform)
                    trans.append(func(transforms=transforms, **parameters))
                else:
                    # load transformation form Albumentations
                    func = getattr(A, transform)
                    trans.append(func(**parameters))
            elif hasattr(A.pytorch, transform):
                # ToTensorV2 transformation is located in A.pytorch
                func = getattr(A.pytorch, transform)
                trans.append(func(**parameters))
            elif hasattr(custom_augmentations, transform):
                func = getattr(custom_augmentations, transform)
                trans.append(func(**parameters))
            else:
                print("No Operation Found: %s", transform)
    return trans


class BaseDataModule(LightningDataModule):
    def __init__(
        self,
        data_input_dir: str,
        dataset,
        batch_size: int,
        val_batch_size: int,
        num_workers: int,
        augmentations: DictConfig,
        tta: bool = False,
        **kwargs,
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
        # self.augmentations = get_augmentations()
        self.augmentations = augmentations
        self.data_input_dir = data_input_dir
        # dataset which is defined in the config
        self.dataset = dataset
        self.test_split = kwargs.get("test_split", None)
        self.tta = tta

    def setup(self, stage: str = None) -> None:
        """
        Setting up the Datasets by initializing the augmentation and the dataloader

        Parameters
        ----------
        stage: str
            current stage which is given by Pytorch Lightning
        """
        if stage in (None, "fit"):
            transforms_train = get_augmentations_from_config(self.augmentations.TRAIN)[
                0
            ]
            self.DS_train = hydra.utils.instantiate(
                self.dataset,
                base_dir=self.data_input_dir,
                split="train",
                transforms=transforms_train,
            )
        if stage in (None, "fit", "validate"):
            transforms_val = get_augmentations_from_config(
                self.augmentations.VALIDATION
            )[0]
            self.DS_val = hydra.utils.instantiate(
                self.dataset,
                base_dir=self.data_input_dir,
                split="val",
                transforms=transforms_val,
                tta=self.tta,
            )
        if stage in (None, "test"):
            transforms_test = get_augmentations_from_config(self.augmentations.TEST)[0]
            test_split = (
                self.test_split
                if self.test_split == "unlabeled" or self.test_split == "val"
                else f"{self.test_split}_test"
            )
            self.DS_test = hydra.utils.instantiate(
                self.dataset,
                base_dir=self.data_input_dir,
                split=test_split,
                transforms=transforms_test,
                tta=self.tta,
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
