import fnmatch
import os
import pickle
import random
from argparse import ArgumentParser
from typing import Optional, List

import numpy as np
import pytorch_lightning as pl
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.utility_transforms import NumpyToTensor
from medpy.io import load
from sklearn.model_selection import KFold


class ToyDataModule3D(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name: str = "Case_1",
        num_raters: int = 3,
        data_input_dir: Optional[str] = None,
        data_num_folds: int = 5,
        data_fold_id: int = 0,
        batch_size: int = 16,
        num_workers: int = 8,
        seed: int = 42,
        *args,
        **kwargs,
    ):
        super(ToyDataModule3D, self).__init__()
        self.dataset_name = dataset_name
        self.num_raters = num_raters
        self.data_input_dir = os.environ.get(
            "DATASET_LOCATION",
            data_input_dir if data_input_dir is not None else os.getcwd(),
        )
        self.data_num_folds = data_num_folds
        self.data_fold_id = data_fold_id
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        # split keys are set in setup()
        self.tr_keys = None
        self.val_keys = None
        self.test_keys = None

    @property
    def num_classes(self):
        return 2

    def prepare_data(self):
        # Preprocess data if this was not already done
        if not os.path.exists(
            os.path.join(self.data_input_dir, self.dataset_name, "preprocessed")
        ):
            print("Preprocessing data. [STARTED]")
            self.preprocess_dataset(
                root_dir=os.path.join(self.data_input_dir, self.dataset_name),
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
                    self.data_input_dir,
                    self.dataset_name,
                    "preprocessed",
                    "imagesTr",
                ),
                seed=self.seed,
                test_dir=os.path.join(
                    self.data_input_dir, self.dataset_name, "preprocessed", "imagesTs"
                ),
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

    def preprocess_dataset(self, root_dir: str):
        """Preprocess the dataset, i.e. normalize the images (z-score normalization) and save

        Args:
            root_dir (str): The root directory of the dataset
        """
        train_test_folder = ["Tr", "Ts"]
        for folder in train_test_folder:
            image_dir = os.path.join(root_dir, "images{}".format(folder))
            label_dir = os.path.join(root_dir, "labels{}".format(folder))
            output_dir_images = os.path.join(
                root_dir, "preprocessed", "images{}".format(folder)
            )
            output_dir_labels = os.path.join(
                root_dir, "preprocessed", "labels{}".format(folder)
            )

            os.makedirs(output_dir_images, exist_ok=True)
            os.makedirs(output_dir_labels, exist_ok=True)

            nii_files = subfiles(image_dir, suffix=".nii.gz", join=False)

            all_images = []
            for f in nii_files:
                image, _ = load(os.path.join(image_dir, f))
                all_images.append(image)
                for rater in range(self.num_raters):
                    label_name = f.replace(
                        ".nii.gz", "_{}.nii.gz".format(str(rater).zfill(2))
                    )
                    label, _ = load(
                        os.path.join(
                            label_dir,
                            label_name,
                        )
                    )
                    np.save(
                        os.path.join(
                            output_dir_labels, label_name.split(".")[0] + ".npy"
                        ),
                        label,
                    )

            for image, f in zip(all_images, nii_files):
                # normalize images
                image = (image - image.mean()) / (
                    image.std() + 1e-8
                )  # TODO using standard z score norm now, keep in mind

                np.save(
                    os.path.join(output_dir_images, f.split(".")[0] + ".npy"), image
                )

    @staticmethod
    def create_splits(output_dir, image_dir, test_dir, seed, n_splits=5):
        """Saves a pickle file containing the splits for k-fold cv on the dataset

        Args:
            output_dir: The output directory where to save the splits file, i.e. the dataset directory
            image_dir: The directory of the preprocessed images
            seed: The seed for the splits
            n_splits: Number of folds
        """
        np.random.seed(seed)
        train_npy_files = subfiles(image_dir, suffix=".npy", join=False, sort=True)
        test_npy_files = subfiles(test_dir, suffix=".npy", join=False, sort=True)

        # array which contains all the splits, one dictionary for each fold
        splits = []

        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        # create fold dictionary and append it to splits
        for i, (train_idx, val_idx) in enumerate(kfold.split(train_npy_files)):
            train_keys = np.array(train_npy_files)[train_idx]
            val_keys = np.array(train_npy_files)[val_idx]
            split_dict = dict()
            split_dict["train"] = train_keys
            split_dict["val"] = val_keys
            split_dict["test"] = np.array(test_npy_files)
            splits.append(split_dict)

        with open(os.path.join(output_dir, "splits.pkl"), "wb") as f:
            pickle.dump(splits, f)

    def train_dataloader(self):
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
            num_raters=self.num_raters,
        )
        train_augmenter = FixedLengthAugmenter(
            data_loader=train_loader,
            transform=Compose([NumpyToTensor(cast_to="float")]),
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
            base_dir=os.path.join(
                self.data_input_dir, self.dataset_name, "preprocessed"
            ),
            batch_size=self.batch_size,
            file_pattern="*.npy",
            subject_ids=self.val_keys,
            num_raters=self.num_raters,
        )
        val_augmenter = FixedLengthAugmenter(
            data_loader=val_loader,
            transform=Compose([NumpyToTensor(cast_to="float")]),
            num_processes=self.num_workers,
            num_cached_per_queue=2,
            seeds=[self.seed] * self.num_workers,
            pin_memory=True,
            timeout=10,
            wait_time=0.02,
        )
        return val_augmenter

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
            "--dataset_name",
            type=str,
            default="Case_1",
            help="Name of the dataset located in the data input directory",
        )
        parser.add_argument(
            "--num_raters",
            type=int,
            default=3,
            help="Number of different segmentations per image",
        )
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
        num_raters: int = 1,
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
            subject_ids=subject_ids,
            num_raters=num_raters,
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
        # slice_idxs = []
        labels = []

        for sample in samples:
            image_array = np.load(sample["image_path"], mmap_mode="r")
            image_array = np.expand_dims(image_array, axis=0)
            # image_slice = np.expand_dims(image_array[sample["slice_idx"]], axis=0)
            data.append(image_array)
            image_paths.append(sample["image_path"])
            if sample["label_path"] is not None:
                label_array = np.load(sample["label_path"], mmap_mode="r")
                label_array = np.expand_dims(label_array, axis=0)
                # slice_label = np.expand_dims(label_array[sample["slice_idx"]], axis=0)
                labels.append(label_array)
                label_paths.append(sample["label_path"])
            # slice_idxs.append(sample["slice_idx"])

        batch = {
            "data": np.asarray(data),
            "image_paths": image_paths,
            "label_paths": label_paths,
            # "slice_idxs": slice_idxs,
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


def get_data_samples(
    base_dir: str,
    pattern: str = "*.npy",
    subject_ids: Optional[List[str]] = None,
    num_raters: int = 1,
    test=False,
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
    train_test = "Tr" if not test else "Ts"
    (image_dir, _, image_filenames) = next(
        os.walk(os.path.join(base_dir, "images{}".format(train_test)))
    )
    (label_dir, _, label_filenames) = next(
        os.walk(os.path.join(base_dir, "labels{}".format(train_test)))
    )
    for image_filename in sorted(fnmatch.filter(image_filenames, pattern)):
        if subject_ids is not None and image_filename in subject_ids:
            image_path = os.path.join(image_dir, image_filename)

            label_paths = []
            for rater in range(num_raters):
                label_path = (
                    os.path.join(
                        label_dir,
                        "{}_{}.npy".format(
                            image_filename.split(".")[0], str(rater).zfill(2)
                        ),
                    )
                    if "{}_{}.npy".format(
                        image_filename.split(".")[0], str(rater).zfill(2)
                    )
                    in label_filenames
                    else None
                )
                if label_path is not None:
                    label_paths.append(label_path)

            label_path = random.choice(label_paths) if len(label_paths) > 0 else None
            samples.append(
                {
                    "image_path": image_path,
                    "label_path": label_path,
                }
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
