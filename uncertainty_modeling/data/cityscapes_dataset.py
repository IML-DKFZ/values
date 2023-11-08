import fnmatch
import os
import pickle

import albumentations
import torch
import numpy as np
import cv2


# cityscapes dataset class
class Cityscapes_dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        splits_path: str,
        base_dir: str,
        split="train",
        file_pattern: str = "*.npy",
        transforms=None,
        data_fold_id: int = 0,
        tta: bool = False,
    ):
        self.splits_path = splits_path
        self.data_fold_id = data_fold_id
        self.get_split_keys()
        if split == "train":
            subject_ids = self.tr_keys
        elif split == "val":
            subject_ids = self.val_keys
        elif split == "id_test":
            subject_ids = self.id_test_keys
        elif split == "ood_test":
            subject_ids = self.ood_test_keys
        elif split == "unlabeled":
            subject_ids = self.unlabeled_keys
        else:
            print(f"{split} split not specified!")

        self.samples = []
        for dataset in ["gta", "cs"]:
            ds_subjects = [
                subject[0] for subject in subject_ids if subject[1] == dataset
            ]
            ds_dir = os.path.join(
                base_dir,
                "OriginalData" if dataset == "gta" else "CityScapesOriginalData",
                "preprocessed",
            )
            self.samples.extend(
                get_data_samples(
                    base_dir=ds_dir,
                    pattern=file_pattern,
                    subject_ids=ds_subjects,
                    dataset=dataset,
                )
            )

        # save all paths in lists
        self.imgs = [sample["image_path"] for sample in self.samples]
        self.masks = [sample["label_path"] for sample in self.samples]
        self.image_ids = [sample["image_id"] for sample in self.samples]
        self.datasets = [sample["dataset"] for sample in self.samples]

        self.transforms = transforms
        self.tta = tta
        print(
            f"Dataset: Cityscape {split} - {len(self.imgs)} images - {len(self.masks)} masks",
        )

    def __getitem__(self, idx):
        # read image (opencv read images in bgr) and mask
        img = np.load(self.imgs[idx])

        mask = np.load(self.masks[idx])

        if self.tta:
            images = [img]
            transforms = [[]]
            flip_transform = albumentations.HorizontalFlip(p=1.0)
            noise_transform = albumentations.GaussNoise(p=1.0)
            flipped = flip_transform(image=img)
            images.append(flipped["image"])
            transforms.append(["HorizontalFlip"])
            noise = noise_transform(image=img)
            images.append(noise["image"])
            transforms.append(["GaussNoise"])
            flipped_noise = noise_transform(image=flipped["image"])
            images.append(flipped_noise["image"])
            transforms.append(["HorizontalFlip", "GaussNoise"])
            images = [self.transforms(image=image)["image"].float() for image in images]
            transformed = self.transforms(image=img, mask=mask)
            mask = transformed["mask"]
            return {
                "data": images,
                "seg": mask,
                "image_id": self.image_ids[idx],
                "dataset": self.datasets[idx],
                "transforms": transforms,
            }  # .long()
        else:
            # apply albumentations transforms
            transformed = self.transforms(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]

            return {
                "data": img.float(),
                "seg": mask,
                "image_id": self.image_ids[idx],
                "dataset": self.datasets[idx],
            }  # .long()

    def __len__(self):
        return len(self.imgs)

    def get_split_keys(self) -> None:
        """Load the keys for training, validation and testing
        Args:
            stage: The current stage of training
        """
        with open(self.splits_path, "rb") as f:
            splits = pickle.load(f)
        self.tr_keys = splits[self.data_fold_id]["train"]
        self.val_keys = splits[self.data_fold_id]["val"]
        self.id_test_keys = splits[self.data_fold_id]["id_test"]
        self.ood_test_keys = splits[self.data_fold_id]["ood_test"]
        self.unlabeled_keys = splits[self.data_fold_id]["id_unlabeled_pool"]
        self.unlabeled_keys = np.concatenate(
            (self.unlabeled_keys, splits[self.data_fold_id]["ood_unlabeled_pool"])
        )


def get_data_samples(
    base_dir: str, pattern: str = "*.npy", subject_ids=None, dataset: str = "gta"
):
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
                    "label_path": label_path,
                    "image_id": image_filename.split(".")[0],
                    "dataset": dataset,
                }
            )

    return samples
