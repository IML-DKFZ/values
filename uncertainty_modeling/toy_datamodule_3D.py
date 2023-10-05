import fnmatch
import os
import pickle
import random
from argparse import ArgumentParser
from typing import Optional, List

import numpy as np
import pytorch_lightning as pl
from batchgenerators.augmentations.crop_and_pad_augmentations import crop
from batchgenerators.augmentations.utils import pad_nd_image
from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.utility_transforms import NumpyToTensor
from batchgenerators.transforms.spatial_transforms import MirrorTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform
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
        patch_size: int = 64,
        patch_overlap: float = 1,
        num_workers: int = 8,
        seed: int = 42,
        augment: bool = False,
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
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.num_workers = num_workers
        self.seed = seed
        self.augment = augment

        # split keys are set in setup()
        self.tr_keys = None
        self.val_keys = None
        self.test_keys = None

    @property
    def num_classes(self) -> int:
        return 2

    def prepare_data(self) -> None:
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

    def setup(self, stage: Optional[str] = None) -> None:
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

    def preprocess_dataset(self, root_dir: str) -> None:
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

            nii_files = subfiles(image_dir, suffix=".nii.gz", prefix="0", join=False)

            all_images = []
            all_labels = []
            for f in nii_files:
                image, _ = load(os.path.join(image_dir, f))
                all_images.append(image)
                raters = []
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
                    raters.append(label)
                all_labels.append(raters)

            for image, label, f in zip(all_images, all_labels, nii_files):
                # normalize images
                image = (image - image.mean()) / (
                    image.std() + 1e-8
                )  # TODO using standard z score norm now, keep in mind
                pad_size_x = image.shape[0] + (
                    image.shape[0] % int(self.patch_size * self.patch_overlap)
                )
                pad_size_y = image.shape[1] + (
                    image.shape[1] % int(self.patch_size * self.patch_overlap)
                )
                pad_size_z = image.shape[2] + (
                    image.shape[2] % int(self.patch_size * self.patch_overlap)
                )
                image = pad_nd_image(
                    image,
                    (pad_size_x, pad_size_y, pad_size_z),
                    "constant",
                    kwargs={"constant_values": image.min()},
                )
                np.save(
                    os.path.join(output_dir_images, f.split(".")[0] + ".npy"), image
                )
                for rater in range(len(label)):
                    label_rater = pad_nd_image(
                        label[rater],
                        (pad_size_x, pad_size_y, pad_size_z),
                        "constant",
                        kwargs={"constant_values": label[rater].min()},
                    )
                    np.save(
                        os.path.join(
                            output_dir_labels,
                            f.split(".")[0] + "_{}.npy".format(str(rater).zfill(2)),
                        ),
                        label_rater,
                    )

    @staticmethod
    def create_splits(output_dir, image_dir, test_dir, seed, n_splits=5) -> None:
        """Saves a pickle file containing the splits for k-fold cv on the dataset

        Args:
            output_dir: The output directory where to save the splits file, i.e. the dataset directory
            image_dir: The directory of the preprocessed training/ validation images
            test_dir: The directory of the preprocessed test images
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
            patch_size=self.patch_size,
        )

        if self.augment:
            transforms = Compose(
                [
                    MirrorTransform(),
                    GaussianNoiseTransform(),
                    NumpyToTensor(cast_to="float"),
                ]
            )
        else:
            transforms = Compose([NumpyToTensor(cast_to="float")])

        train_augmenter = FixedLengthAugmenter(
            data_loader=train_loader,
            transform=transforms,
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
            batch_size=1,
            file_pattern="*.npy",
            subject_ids=self.val_keys,
            num_raters=self.num_raters,
            training=False,
            patch_size=self.patch_size,
        )
        val_augmenter = FixedLengthAugmenter(
            data_loader=val_loader,
            transform=Compose([NumpyToTensor(cast_to="float")]),
            num_processes=1,
            num_cached_per_queue=2,
            seeds=[self.seed],
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
        parser.add_argument(
            "--patch_size",
            type=int,
            default=64,
            help="Patch size.",
        )
        parser.add_argument(
            "--patch_overlap",
            type=float,
            default=1,
            help="Ratio of patch overlap. 1 means no patch overlap, 0.5 half the patch size overlap etc.",
        )
        return parser


class NumpyDataLoader(DataLoader):
    def __init__(
        self,
        base_dir: str,
        batch_size: int = 16,
        patch_size: int = 64,
        patch_overlap: float = 1,
        file_pattern: str = "*.npy",
        subject_ids: Optional[List[str]] = None,
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
        """
        if training:
            self.samples = get_train_data_samples(
                base_dir=base_dir,
                pattern=file_pattern,
                subject_ids=subject_ids,
                num_raters=num_raters,
            )
        # Validation loop
        else:
            self.samples = get_val_test_data_samples(
                base_dir=base_dir,
                pattern=file_pattern,
                subject_ids=subject_ids,
                num_raters=num_raters,
                test=False,
                patch_size=patch_size,
                patch_overlap=patch_overlap,
            )
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
            image_array = np.load(sample["image_path"], mmap_mode="r")
            # Add channel to image dimension
            image_array = np.expand_dims(image_array, axis=0)
            if self.training:
                # Second expand is needed for batchgenerators crop
                image_array = np.expand_dims(image_array, axis=0)
            image_paths.append(sample["image_path"])
            if sample["label_paths"] is not None:
                label_path = random.choice(sample["label_paths"])
                label_array = np.load(label_path, mmap_mode="r")
                # Add channel to image dimension
                label_array = np.expand_dims(label_array, axis=0)
                if self.training:
                    # Second expand is needed for batchgenerators crop
                    label_array = np.expand_dims(label_array, axis=0)
                label_paths.append(label_path)
            else:
                label_array = None
            if self.training:
                image_patch, label_patch = crop(
                    image_array, label_array, self.patch_size, crop_type="random"
                )
                image_patch = image_patch[0]
                if label_patch is not None:
                    label_patch = label_patch[0]
            else:
                image_patch = image_array[
                    :,
                    sample["crop_idx"][0][0] : sample["crop_idx"][0][1],
                    sample["crop_idx"][1][0] : sample["crop_idx"][1][1],
                    sample["crop_idx"][2][0] : sample["crop_idx"][2][1],
                ]
                if label_array is not None:
                    label_patch = label_array[
                        :,
                        sample["crop_idx"][0][0] : sample["crop_idx"][0][1],
                        sample["crop_idx"][1][0] : sample["crop_idx"][1][1],
                        sample["crop_idx"][2][0] : sample["crop_idx"][2][1],
                    ]
                else:
                    label_patch = None
            data.append(image_patch)
            labels.append(label_patch)

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


def get_train_data_samples(
    base_dir: str,
    pattern: str = "*.npy",
    subject_ids: Optional[List[str]] = None,
    num_raters: int = 1,
) -> List[dict]:
    """
    Return a list of all possible input samples in the dataset by returning all possible slices for each subject id.

    Args:
        base_dir (str): Directory where preprocessed numpy files reside. Should contain subfolders imagesTr and labelsTr
        pattern (str): Pattern to match os.walk filenames against.
        subject_ids (list/array): Which subject IDs to load.
        num_raters (int): Number of segmentation variants
        test (bool): Whether testing or training is done

    Returns:
        samples [List[dict]]: All possible slices for each subject id.
    """
    samples = []
    (image_dir, _, image_filenames) = next(os.walk(os.path.join(base_dir, "imagesTr")))
    (label_dir, _, label_filenames) = next(os.walk(os.path.join(base_dir, "labelsTr")))
    for image_filename in sorted(fnmatch.filter(image_filenames, pattern)):
        if subject_ids is None or image_filename in subject_ids:
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

            label_paths = label_paths if len(label_paths) > 0 else None
            samples.append(
                {
                    "image_path": image_path,
                    "label_paths": label_paths,
                }
            )

    return samples


def get_val_test_data_samples(
    base_dir: str,
    pattern: str = "*.npy",
    subject_ids: Optional[List[str]] = None,
    num_raters: int = 1,
    test=False,
    patch_size: int = 64,
    patch_overlap: float = 1,
) -> List[dict]:
    """
    Return a list of all possible input samples in the dataset by returning all possible slices for each subject id.

    Args:
        base_dir (str): Directory where preprocessed numpy files reside. Should contain subfolders imagesTr and labelsTr
        pattern (str): Pattern to match os.walk filenames against.
        subject_ids (list/array): Which subject IDs to load.
        num_raters (int): Number of segmentation variants
        test (bool): Whether testing or training is done

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
        if subject_ids is None or image_filename in subject_ids:
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

            label_paths = label_paths if len(label_paths) > 0 else None

            image_array = np.load(image_path, mmap_mode="r")

            start_idx_z = 0

            crop_indices = []
            while start_idx_z <= image_array.shape[2] - patch_size:
                start_idx_y = 0
                while start_idx_y <= image_array.shape[1] - patch_size:
                    start_idx_x = 0
                    while start_idx_x <= image_array.shape[0] - patch_size:
                        crop_indices.append(
                            (
                                (start_idx_x, start_idx_x + patch_size),
                                (start_idx_y, start_idx_y + patch_size),
                                (start_idx_z, start_idx_z + patch_size),
                            )
                        )
                        start_idx_x += int(patch_size * patch_overlap)
                    start_idx_y += int(patch_size * patch_overlap)
                start_idx_z += int(patch_size * patch_overlap)

            for crop_index in crop_indices:
                samples.append(
                    {
                        "image_path": image_path,
                        "label_paths": label_paths,
                        "crop_idx": crop_index,
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
