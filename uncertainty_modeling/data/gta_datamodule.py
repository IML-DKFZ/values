import fnmatch
import os
import pickle

import numpy as np
import pytorch_lightning as pl
from typing import Optional, List, Tuple

from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import Compose, AbstractTransform
from batchgenerators.transforms.utility_transforms import NumpyToTensor

import cityscapes_labels as cs_labels


class GTADataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_input_dir: Optional[str] = None,
        num_classes: int = 25,
        data_fold_id: int = 0,
        batch_size: int = 16,
        patch_size: Tuple[int, int] = (478, 256),
        num_workers: int = 8,
        seed: int = 42,
        splits_path: str = None,
        *args,
        **kwargs
    ):
        super(GTADataModule, self).__init__()
        self.data_input_dir = os.environ.get(
            "DATASET_LOCATION",
            data_input_dir if data_input_dir is not None else os.getcwd(),
        )
        self.data_fold_id = data_fold_id
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.seed = seed
        if splits_path is not None:
            self.splits_path = splits_path
        else:
            self.splits_path = os.path.join(self.data_input_dir, "splits.pkl")

        # split keys are set in setup()
        self.tr_keys = None
        self.val_keys = None
        self.test_keys = None

    @property
    def num_classes(self) -> int:
        return 25

    def setup(self, stage: Optional[str] = None) -> None:
        """Load the keys for training, validation and testing
        Args:
            stage: The current stage of training
        """
        with open(self.splits_path, "rb") as f:
            splits = pickle.load(f)
        self.tr_keys = splits[self.data_fold_id]["train"]
        self.val_keys = splits[self.data_fold_id]["val"]
        self.test_keys = splits[self.data_fold_id]["test"]

    def train_dataloader(self):
        """Dataloader for training. Loads numpy data and defines transformations

        Returns:
            [FixedLengthAugmenter]: Multithreaded train dataloader with augmentations
        """
        train_loader = NumpyDataLoader(
            base_dir=os.path.join(self.data_input_dir, "preprocessed"),
            batch_size=self.batch_size,
            file_pattern="*.npy",
            subject_ids=self.tr_keys,
        )

        transforms = [StochasticLabelSwitches(), NumpyToTensor(cast_to="float")]
        train_augmenter = FixedLengthAugmenter(
            data_loader=train_loader,
            transform=Compose(transforms),
            num_processes=self.num_workers,
            num_cached_per_queue=2,
            seeds=[self.seed] * self.num_workers,
            pin_memory=True,
            timeout=10,
            wait_time=0.02,
        )
        return train_augmenter

    def val_dataloader(self):
        """Dataloader for validation. Loads numpy data and defines transformations

        Returns:
            [FixedLengthAugmenter]: Multithreaded validation dataloader with augmentations
        """
        val_loader = NumpyDataLoader(
            base_dir=os.path.join(self.data_input_dir, "preprocessed"),
            batch_size=1,
            file_pattern="*.npy",
            subject_ids=self.val_keys,
            training=False,
        )

        transforms = [StochasticLabelSwitches(), NumpyToTensor(cast_to="float")]
        val_augmenter = FixedLengthAugmenter(
            data_loader=val_loader,
            transform=Compose(transforms),
            num_processes=1,
            num_cached_per_queue=2,
            seeds=[self.seed],
            pin_memory=True,
            timeout=10,
            wait_time=0.02,
        )
        return val_augmenter


class NumpyDataLoader(DataLoader):
    def __init__(
        self,
        base_dir: str,
        batch_size: int = 16,
        patch_size: int = 64,
        file_pattern: str = "*.npy",
        subject_ids: Optional[List[str]] = None,
        training: bool = True,
    ):
        """
        Subclass of batchgenerators' SlimDataLoaderBase to sample from possible subject/slice combinations given by
        get_data_samples.

        Args:
            base_dir (str): Directory where preprocessed numpy files reside.
                            Should contain subfolders imagesTr and labelsTr.
            batch_size (int): Batch size.
            file_pattern (str): Pattern to match os.walk filenames against when resolving possible subjects/slices.
            subject_ids (list/array): Which subject IDs to load.
        """
        self.samples = []
        for dataset in ["gta", "cs"]:
            ds_subjects = [
                subject[0] for subject in subject_ids if subject[1] == dataset
            ]
            ds_dir = os.path.join(
                base_dir,
                "OriginalData" if dataset == "gta" else "CityScapesOriginalData",
            )
            self.samples.extend(
                get_data_samples(
                    base_dir=ds_dir, pattern=file_pattern, subject_ids=ds_subjects
                )
            )
        # self.samples = get_data_samples(
        #     base_dir=base_dir, pattern=file_pattern, subject_ids=subject_ids
        # )
        super(NumpyDataLoader, self).__init__(self.samples, batch_size)

        self.num_restarted = 0
        self.current_position = 0
        self.was_initialized = False
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.training = training
        self.indices = list(range(len(self.samples)))

    def reset(self):
        # Prevents the random order for each epoch being the same
        rs = np.random.RandomState(self.num_restarted)

        # Here the data is shuffled
        if self.training:
            rs.shuffle(self._data)
        self.was_initialized = True
        self.num_restarted = self.num_restarted + 1

        # Select a starting point for this subprocess
        self.current_position = self.thread_id * self.batch_size

    def generate_train_batch(self) -> dict:
        """Generate batch. Overwritten function from batchgenerators' SlimDataLoaderBase.
        Returns:
            batch (dict): The generated batch
        """
        # For doing the initialization in each subprocess generated by MultiThreadedAugmentor
        if not self.was_initialized:
            self.reset()
        # This will be used for the batch starting point in this loop
        idx = self.current_position

        if idx < len(self._data):
            # Next starting point. This skips the length of one batch for
            # this process AS WELL AS all the other processes
            self.current_position = (
                idx + self.batch_size * self.number_of_threads_in_multithreaded
            )
            samples = self._data[idx : min(len(self._data), idx + self.batch_size)]
        else:
            self.was_initialized = False
            return self.generate_train_batch()

        data = []
        image_paths = []
        label_paths = []
        labels = []

        for sample in samples:
            image_array = np.load(sample["image_path"], mmap_mode="r") / 255.0
            # TODO: probably do this in preprocessing
            # input is NCHW
            image_array = np.transpose(image_array, axes=[2, 0, 1])
            # TODO: Probably not needed because we already have 3 channels
            # Add channel to image dimension
            # image_array = np.expand_dims(image_array, axis=0)
            image_paths.append(sample["image_path"])
            if sample["label_paths"] is not None:
                label_array = np.load(sample["label_path"], mmap_mode="r")
                # Add channel to image dimension
                label_array = np.expand_dims(label_array, axis=0)
                label_paths.append(sample["label_path"])
            else:
                label_array = None
            data.append(image_array)
            labels.append(label_array)

        batch = {
            "data": np.asarray(data),
            "image_paths": image_paths,
            "label_paths": label_paths,
        }
        if len(labels) > 0:
            batch["seg"] = np.asarray(labels)

        return batch

    def __len__(self):
        return len(self.samples)


class FixedLengthAugmenter(MultiThreadedAugmenter):
    """Subclass of batchgenerators' MultiThreadedAugmenter to enable multithreaded dataloading"""

    def __len__(self):
        return len(self.generator)


class StochasticLabelSwitches(AbstractTransform):
    """
    Stochastically switches labels in a batch of integer-labeled segmentations.
    """

    def __init__(self):
        self._name2id = cs_labels.name2trainId
        self._label_switches = {
            "sidewalk": 1.0 / 2.0,
            "person": 1.0 / 2.0,
            "car": 1.0 / 2.0,
            "vegetation": 1.0 / 2.0,
            "road": 1.0 / 2.0,
        }

    def __call__(self, **data_dict):

        switched_seg = data_dict["seg"]
        batch_size = switched_seg.shape[0]

        for c, p in self._label_switches.items():
            init_id = self._name2id[c]
            final_id = self._name2id[c + "_2"]
            switch_instances = np.random.binomial(1, p, batch_size)

            for i in range(batch_size):
                if switch_instances[i]:
                    switched_seg[i][switched_seg[i] == init_id] = final_id

        data_dict["seg"] = switched_seg
        return data_dict


def get_data_samples(
    base_dir: str,
    pattern: str = "*.npy",
    subject_ids: Optional[List[str]] = None,
) -> List[dict]:
    """
    Return a list of all possible input samples in the dataset by returning all possible slices for each subject id.

    Args:
        base_dir (str): Directory where preprocessed numpy files reside. Should contain subfolders imagesTr and labelsTr
        pattern (str): Pattern to match os.walk filenames against.
        subject_ids (list/array): Which subject IDs to load.

    Returns:
        samples [List[dict]]: All possible slices for each subject id.
    """
    samples = []

    (image_dir, _, image_filenames) = next(os.walk(os.path.join(base_dir, "images")))
    (label_dir, _, label_filenames) = next(os.walk(os.path.join(base_dir, "labels")))
    for image_filename in sorted(fnmatch.filter(image_filenames, pattern)):
        if subject_ids is not None and image_filename in subject_ids:
            image_path = os.path.join(image_dir, image_filename)

            label_path = (
                os.path.join(label_dir, image_filename)
                if image_filename in label_filenames
                else None
            )

            samples.append(
                {
                    "image_path": image_path,
                    "label_paths": label_path,
                }
            )

    return samples
