from __future__ import annotations

import os
from operator import itemgetter
from typing import Optional, List
import random
import pickle
import tarfile
import fnmatch
import numpy as np
import pytorch_lightning as pl
from sklearn.model_selection import KFold
from argparse import ArgumentParser

from google_drive_downloader import GoogleDriveDownloader as Gdd

from medpy.io import load
from batchgenerators.augmentations.utils import pad_nd_image
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.spatial_transforms import (
    MirrorTransform,
    ResizeTransform,
    SpatialTransform,
)
from batchgenerators.transforms.utility_transforms import NumpyToTensor
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter


class MSDDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        data_input_dir: Optional[str] = None,
        data_num_folds: int = 5,
        data_fold_id: int = 0,
        batch_size: int = 16,
        y_shape: int = None,
        z_shape: int = None,
        num_workers: int = 8,
        seed: int = 42,
        google_drive_id: str = None,
        *args,
        **kwargs,
    ):
        """
        Pytorch-Lightning Base DataModule for Medical Segmentation Decathlon Tasks,
        alongside data transformation, loading & preprocessing.
        Note, that currently the google drive downloader might fail for large datasets of the MSD.
        In this case, they need to be downloaded manually.

        Args:
            dataset_name (str): The name of the MSD task usually named Task<number>_<Name>
                                just like the zip files of the MSD on Google Drive.
            data_input_dir (str): Where to save/load the dataset.
            data_num_folds (int): How many folds to split the dataset into. Creates splits.pkl file if not yet present.
            data_fold_id (int): Which fold to use.
            batch_size (int): Which batch size to use for dataloaders.
            num_workers (int): How many workers to use for loading data. Please set according to your CPU.
            google_drive_id (str): The google drive id to download the data. If None, the data is expected
                                    to be downloaded manually.
            y_shape (int): Desired y shape of the data used for padding.
                           If None, the maximum y shape across all samples is taken
            z_shape (int): Desired z shape of the data used for padding.
                           If None, the maximum y shape across all samples is taken
        """
        super(MSDDataModule, self).__init__()
        self.dataset_name = dataset_name
        self.data_input_dir = os.environ.get(
            "DATASET_LOCATION",
            data_input_dir if data_input_dir is not None else os.getcwd(),
        )
        self.data_num_folds = data_num_folds
        self.data_fold_id = data_fold_id
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.google_drive_id = google_drive_id
        self.y_shape = y_shape
        self.z_shape = z_shape

        # split keys are set in setup()
        self.tr_keys = None
        self.val_keys = None
        self.test_keys = None

    @property
    def num_classes(self):
        return 3

    def prepare_data(self):
        """Downloads the data, preprocesses it and creates split file if not existing"""

        # Download dataset
        if self.google_drive_id is not None:
            self.download_dataset(
                dest_path=self.data_input_dir,
                dataset=self.dataset_name,
                google_drive_id=self.google_drive_id,
            )
        else:
            assert not os.path.exists(
                os.path.join(self.data_input_dir, self.dataset_name)
            ), "Data is not downloaded in the expected location and you did not specify a google drive id to download"

        # Preprocess data if this was not already done
        if not os.path.exists(
            os.path.join(self.data_input_dir, self.dataset_name, "preprocessed")
        ):
            print("Preprocessing data. [STARTED]")
            self.preprocess_dataset(
                root_dir=os.path.join(self.data_input_dir, self.dataset_name),
                y_shape=self.y_shape,
                z_shape=self.z_shape,
            )
            print("Preprocessing data. [DONE]")
        else:
            print(
                "Data already preprocessed. Data is not preprocessed again. Delete folder to enforce it."
            )

        # Create splits.pkl file if this is not already existing to define the splits
        if not os.path.exists(
            os.path.join(self.data_input_dir, self.dataset_name, "splits.pkl")
        ):
            print(
                "No splits pickle file found."
                "Creating new splits file for {} fold cross-validation.".format(
                    self.data_num_folds
                )
            )
            self.create_splits(
                output_dir=os.path.join(self.data_input_dir, self.dataset_name),
                image_dir=os.path.join(
                    self.data_input_dir, self.dataset_name, "preprocessed", "imagesTr"
                ),
                seed=self.seed,
                fraction_test=0.25,
                n_splits=self.data_num_folds,
            )
        else:
            print("Splits files found. Using the folds specified in the file.")

    def setup(self, stage: Optional[str] = None):
        """Load the keys for training, validation and testing
        Args:
            stage: The current stage of training
        """
        with open(
            os.path.join(self.data_input_dir, self.dataset_name, "splits.pkl"), "rb"
        ) as f:
            splits = pickle.load(f)
        self.tr_keys = splits[self.data_fold_id]["train"]
        self.val_keys = splits[self.data_fold_id]["val"]
        self.test_keys = splits[self.data_fold_id]["test"]

    @staticmethod
    def download_dataset(dest_path: str, dataset: str, google_drive_id: str = ""):
        """Downloads dataset from Google Drive and stores it in <dest_path>/<dataset>

        Args:
            dest_path (str): The root path to store the datasets
            dataset (str): The dataset name
            google_drive_id (str): The file id to download from Google drive
        """
        if not os.path.exists(os.path.join(dest_path, dataset)):
            tar_path = os.path.join(dest_path, dataset, dataset) + ".tar"
            Gdd.download_file_from_google_drive(
                file_id=google_drive_id,
                dest_path=tar_path,
                overwrite=False,
                unzip=False,
            )

            print("Extracting data [STARTED]")
            tar = tarfile.open(tar_path)
            tar.extractall(dest_path)
            print("Extracting data [DONE]")
        else:
            print("Data already downloaded. Files are not extracted again.")

    @staticmethod
    def preprocess_dataset(root_dir: str, y_shape: int = None, z_shape: int = None):
        """Preprocess the dataset, i.e. normalize the images (z-score normalization) and pad the y and z dimension

        Args:
            root_dir (str): The root directory of the dataset
            y_shape (int, Optional): y shape for padding
            z_shape (int, Optional): z shape for padding
        """
        image_dir = os.path.join(root_dir, "imagesTr")
        label_dir = os.path.join(root_dir, "labelsTr")
        output_dir_images = os.path.join(root_dir, "preprocessed", "imagesTr")
        output_dir_labels = os.path.join(root_dir, "preprocessed", "labelsTr")

        os.makedirs(output_dir_images, exist_ok=True)
        os.makedirs(output_dir_labels, exist_ok=True)

        nii_files = subfiles(image_dir, suffix=".nii.gz", join=False)

        # legacy: specific for local file format
        for i in range(0, len(nii_files)):
            if nii_files[i].startswith("._"):
                nii_files[i] = nii_files[i][2:]

        all_images = []
        all_labels = []
        for f in nii_files:
            image, _ = load(os.path.join(image_dir, f))
            all_images.append(image)
            # legacy: specific for nnU-Net format
            label, _ = load(os.path.join(label_dir, f.replace("_0000", "")))
            all_labels.append(label)

        image_shapes = [image.shape for image in all_images]
        if y_shape is None:
            y_shape = max(image_shapes, key=itemgetter(1))[1]
        if z_shape is None:
            z_shape = max(image_shapes, key=itemgetter(2))[2]
        # TODO: quadratic padding for network input with resize transform, maybe change later
        y_shape = max(y_shape, z_shape)
        z_shape = max(y_shape, z_shape)
        print("Using {} as z_shape for padding".format(z_shape))
        print("Using {} as y_shape for padding".format(y_shape))
        for image, label, f in zip(all_images, all_labels, nii_files):
            # normalize images
            # image = (image - image.min()) / (image.max() - image.min())
            image = (image - image.mean()) / (
                image.std() + 1e-8
            )  # TODO using standard z score norm now, keep in mind

            # TODO does this result in images with different x-shapes? (if original images have different x-dims)
            # Is image.shape[0] actually transversal (slice) direction?
            image = pad_nd_image(
                image,
                (image.shape[0], y_shape, z_shape),
                "constant",
                kwargs={"constant_values": image.min()},
            )
            label = pad_nd_image(
                label,
                (image.shape[0], y_shape, z_shape),
                "constant",
                kwargs={"constant_values": label.min()},
            )

            np.save(os.path.join(output_dir_images, f.split(".")[0] + ".npy"), image)
            np.save(os.path.join(output_dir_labels, f.split(".")[0] + ".npy"), label)

    @staticmethod
    def create_splits(output_dir, image_dir, seed, fraction_test, n_splits=5):
        """Saves a pickle file containing the splits for k-fold cv on the dataset

        Args:
            output_dir: The output directory where to save the splits file, i.e. the dataset directory
            image_dir: The directory of the preprocessed images
            seed: The seed for the splits
            fraction_test: Fraction of the testset
            n_splits: Number of folds
        """
        np.random.seed(seed)
        npy_files = subfiles(image_dir, suffix=".npy", join=False)

        # take out a testset that is the same for all splits
        testset_size = int(len(npy_files) * fraction_test)
        test_keys = []
        for i in range(0, testset_size):
            patient = np.random.choice(npy_files)
            npy_files.remove(patient)
            test_keys.append(patient)

        # array which contains all the splits, one dictionary for each fold
        splits = []
        all_files_sorted = np.sort(npy_files)

        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        # create fold dictionary and append it to splits
        for i, (train_idx, val_idx) in enumerate(kfold.split(all_files_sorted)):
            train_keys = np.array(all_files_sorted)[train_idx]
            val_keys = np.array(all_files_sorted)[val_idx]
            split_dict = dict()
            split_dict["train"] = train_keys
            split_dict["val"] = val_keys
            split_dict["test"] = np.array(test_keys)
            splits.append(split_dict)

        with open(os.path.join(output_dir, "splits.pkl"), "wb") as f:
            pickle.dump(splits, f)

    @staticmethod
    def get_transforms(mode: str = "train", target_size: int = 128) -> Compose:
        """Define the desired transformation for the data using batchgenerators

        Args:
            mode: The mode for the predictions ("train", "val" or "test").
                  In training mode more transformations are applied than in validation/ testing
            target_size: Size to which the image should be resized

        Returns:
            [Compose]: All the transformations that should be applied to the image
        """
        transform_list = []

        if mode == "train":
            transform_list = [
                ResizeTransform(target_size=(target_size, target_size), order=1),
                MirrorTransform(axes=(1,)),
                SpatialTransform(
                    patch_size=(target_size, target_size),
                    random_crop=False,
                    patch_center_dist_from_border=target_size // 2,
                    do_elastic_deform=True,
                    alpha=(0.0, 900.0),
                    sigma=(20.0, 30.0),
                    do_rotation=True,
                    p_rot_per_sample=0.8,
                    angle_x=(-15.0 / 360 * 2.0 * np.pi, 15.0 / 360 * 2.0 * np.pi),
                    angle_y=(0, 1e-8),
                    angle_z=(0, 1e-8),
                    scale=(0.85, 1.25),
                    p_scale_per_sample=0.8,
                    border_mode_data="nearest",
                    border_mode_seg="nearest",
                ),
            ]

        elif mode == "val" or mode == "test":
            transform_list = [
                ResizeTransform(target_size=(target_size, target_size), order=1),
            ]

        transform_list.append(NumpyToTensor())

        return Compose(transform_list)

    def train_dataloader(self) -> FixedLengthAugmenter:
        """Dataloader for training. Loads numpy data and defines transformations

        Returns:
            [FixedLengthAugmenter]: Multithreaded train dataloader with augmentations
        """
        train_loader = NumpyDataLoader(
            base_dir=os.path.join(
                self.data_input_dir, self.dataset_name, "preprocessed"
            ),
            batch_size=self.batch_size,
            file_pattern="*.npy",
            subject_ids=self.tr_keys,
        )
        train_augmenter = FixedLengthAugmenter(
            data_loader=train_loader,
            transform=self.get_transforms(mode="train", target_size=128),
            num_processes=self.num_workers,
            num_cached_per_queue=2,
            seeds=[self.seed] * self.num_workers,
            pin_memory=True,
            timeout=10,
            wait_time=0.02,
        )
        return train_augmenter

    def val_dataloader(self) -> FixedLengthAugmenter:
        """Dataloader for validation. Loads numpy data and defines transformations

        Returns:
            [FixedLengthAugmenter]: Multithreaded validation dataloader with augmentations
        """
        val_loader = NumpyDataLoader(
            base_dir=os.path.join(
                self.data_input_dir, self.dataset_name, "preprocessed"
            ),
            batch_size=self.batch_size,
            file_pattern="*.npy",
            subject_ids=self.val_keys,
        )
        val_augmenter = FixedLengthAugmenter(
            data_loader=val_loader,
            transform=self.get_transforms(mode="val", target_size=128),
            num_processes=self.num_workers,
            num_cached_per_queue=2,
            seeds=[self.seed] * self.num_workers,
            pin_memory=True,
            timeout=10,
            wait_time=0.02,
        )
        return val_augmenter

    def test_dataloader(self) -> TestAugmenter:
        """Dataloader for testing. Loads numpy data and defines transformations

        Returns:
            [FixedLengthAugmenter]: Multithreaded test dataloader with augmentations
        """
        test_loader = NumpyDataLoader(
            base_dir=os.path.join(
                self.data_input_dir, self.dataset_name, "preprocessed"
            ),
            batch_size=self.batch_size,
            file_pattern="*.npy",
            subject_ids=self.test_keys,
            training=False,
        )
        test_augmenter = TestAugmenter(
            data_loader=test_loader,
            transform=self.get_transforms(mode="test", target_size=128),
            num_processes=1,
            num_cached_per_queue=2,
            seeds=[self.seed],
            pin_memory=True,
            timeout=10,
            wait_time=0.02,
        )
        return test_augmenter

    @staticmethod
    def add_data_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """Add arguments to parser that are specific for datamodule
           (data input directory, number of folds, fold id, number of workers for dataloader, batch size)

        Args:
            parent_parser (ArgumentParser): The parser to add the arguments

        Returns:
            parser [ArgumentParser]: The parent parser with the appended module specific arguments
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "-i",
            "--data_input_dir",
            type=str,
            default="/home/finja/Documents/Datasets/MSD",
            help="Dataset root directory to load from.",
        )
        parser.add_argument(
            "--data_num_folds",
            type=int,
            default=5,
            help="Number of folds for K-fold cross-validation.",
        )
        parser.add_argument(
            "--data_fold_id",
            type=int,
            default=0,
            help="Which fold in K-fold cross-validation to use for train/val set.",
        )
        parser.add_argument(
            "--num_workers",
            type=int,
            default=8,
            help="Number of workers used for the data loader.",
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            default=16,
            help="Batch size.",
        )
        return parser


class NumpyDataLoader(SlimDataLoaderBase):
    def __init__(
        self,
        base_dir: str,
        batch_size: int = 16,
        file_pattern: str = "*.npy",
        subject_ids: Optional[List[str]] = None,
        slice_offset: int = 0,
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
            slice_offset (int): The offset that is not considered for the 2D slices
        """
        self.samples = get_data_samples(
            base_dir=base_dir,
            pattern=file_pattern,
            slice_offset=slice_offset,
            subject_ids=subject_ids,
        )
        super(NumpyDataLoader, self).__init__(self.samples, batch_size)

        self.batch_size = batch_size
        self.training = training
        self.start_idx = 0

    def generate_train_batch(self) -> dict:
        """Generate batch. Overwritten function from batchgenerators' SlimDataLoaderBase.
        Returns:
            batch (dict): The generated batch
        """
        if self.training:
            samples = random.sample(self._data, self.batch_size)
        else:
            samples = [
                x
                for idx, x in enumerate(self._data)
                if idx >= self.start_idx and idx < self.start_idx + self.batch_size
            ]
            self.start_idx += self.batch_size

        data = []
        image_paths = []
        label_paths = []
        slice_idxs = []
        labels = []

        for sample in samples:
            image_array = np.load(sample["image_path"], mmap_mode="r")
            image_slice = np.expand_dims(image_array[sample["slice_idx"]], axis=0)
            data.append(image_slice)
            image_paths.append(sample["image_path"])
            if sample["label_path"] is not None:
                label_array = np.load(sample["label_path"], mmap_mode="r")
                slice_label = np.expand_dims(label_array[sample["slice_idx"]], axis=0)
                labels.append(slice_label)
                label_paths.append(sample["label_path"])
            slice_idxs.append(sample["slice_idx"])

        batch = {
            "data": np.asarray(data),
            "image_paths": image_paths,
            "label_paths": label_paths,
            "slice_idxs": slice_idxs,
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


class TestAugmenter(MultiThreadedAugmenter):
    """Subclass of batchgenerators' MultiThreadedAugmenter to enable multithreaded dataloading"""

    def __len__(self):
        return int((len(self.generator) / self.generator.batch_size) - 1)


# TODO: probably remove here (also in save_predicticns)
def get_data_samples(
    base_dir: str,
    pattern: str = "*.npy",
    slice_offset: int = 5,
    subject_ids: Optional[List[str]] = None,
) -> List[dict]:
    """
    Return a list of all possible input samples in the dataset by returning all possible slices for each subject id.

    Args:
        base_dir (str): Directory where preprocessed numpy files reside. Should contain subfolders imagesTr and labelsTr
        pattern (str): Pattern to match os.walk filenames against.
        slice_offset (int): Offset possible slices in the dataset.
        subject_ids (list/array): Which subject IDs to load.

    Returns:
        samples [List[dict]]: All possible slices for each subject id.
    """
    samples = []

    (image_dir, _, image_filenames) = next(os.walk(os.path.join(base_dir, "imagesTr")))
    (label_dir, _, label_filenames) = next(os.walk(os.path.join(base_dir, "labelsTr")))
    for image_filename in sorted(fnmatch.filter(image_filenames, pattern)):
        if subject_ids is not None and image_filename in subject_ids:
            image_path = os.path.join(image_dir, image_filename)
            image_array = np.load(image_path, mmap_mode="r")
            file_len = image_array.shape[0]

            label_path = (
                os.path.join(label_dir, image_filename)
                if image_filename in label_filenames
                else None
            )

            samples.extend(
                [
                    {"image_path": image_path, "label_path": label_path, "slice_idx": i}
                    for i in range(slice_offset, file_len - slice_offset)
                ]
            )

    return samples


def subfiles(
    folder: str,
    join: bool = True,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    sort: bool = True,
) -> List[str]:
    """Return all the files that are in a specific directory: Possible to filter for certain prefix, suffix and to sort
    Args:
        folder (str): The folder in which the files are
        join (bool): Whether to return path with folder or only filename
        prefix (str, optional): Prefix to filter the files. Only files with this prefix will be returned.
        suffix (str, optional): Suffix to filter the files. Only files with this suffix will be returned.
        sort (bool): Whether to sort the files.

    Returns:
        res [List[str]]: List with filtered files of the directory
    """
    if join:
        l = os.path.join
    else:
        l = lambda x, y: y
    res = [
        l(folder, i)
        for i in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, i))
        and (prefix is None or i.startswith(prefix))
        and (suffix is None or i.endswith(suffix))
    ]
    if sort:
        res.sort()
    return res
