import fnmatch
import os
import pickle

import torch
import numpy as np
import cv2


# cityscapes dataset class
class Cityscapes_dataset(torch.utils.data.Dataset):
    def __init__(
        self, splits_path: str, data_input_dir: str, split="train", transforms=None
    ):
        self.splits_path = splits_path
        self.get_split_keys()
        if split == "train":
            subject_ids = self.tr_keys
        elif split == "val":
            subject_ids = self.val_keys
        else:
            subject_ids = self.id_test_keys

        ds_subjects = [subject[0] for subject in subject_ids if subject[1] == "cs"]
        ds_dir = os.path.join(
            data_input_dir,
            "CityScapesOriginalData",
            "preprocessed",
        )

        self.samples = get_data_samples(
            base_dir=ds_dir, pattern="*.npy", subject_ids=ds_subjects
        )

        # save all paths in lists
        self.imgs = [sample["image_path"] for sample in self.samples]
        self.masks = [sample["label_path"] for sample in self.samples]

        self.transforms = transforms
        print(
            f"Dataset: Cityscape {split} - {len(self.imgs)} images - {len(self.masks)} masks",
        )

    def __getitem__(self, idx):
        # read image (opencv read images in bgr) and mask
        img = np.load(self.imgs[idx])

        mask = np.load(self.masks[idx])
        # img = cv2.imread(self.imgs[idx])
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #
        # mask = cv2.imread(self.masks[idx], -1)

        # apply albumentations transforms
        transformed = self.transforms(image=img, mask=mask)
        img = transformed["image"]
        mask = transformed["mask"]

        return {"data": img.float(), "seg": mask}  # .long()

    def __len__(self):
        return len(self.imgs)

    def get_split_keys(self) -> None:
        """Load the keys for training, validation and testing
        Args:
            stage: The current stage of training
        """
        with open(self.splits_path, "rb") as f:
            splits = pickle.load(f)
        self.data_fold_id = 0
        self.tr_keys = splits[self.data_fold_id]["train"]
        self.val_keys = splits[self.data_fold_id]["val"]
        self.id_test_keys = splits[self.data_fold_id]["id_test"]
        self.ood_test_keys = splits[self.data_fold_id]["ood_test"]


def get_data_samples(
    base_dir: str,
    pattern: str = "*.npy",
    subject_ids=None,
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
    # subject_ids = [subject.replace(".npy", ".png") for subject in subject_ids]
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
                }
            )

    return samples
