import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List

import numpy as np
from batchgenerators.augmentations.utils import pad_nd_image
from batchgenerators.utilities.file_and_folder_operations import subfiles
from medpy.io import load


def main_cli():
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        "-d",
        type=str,
        help="Root path to the original dataset with images and labels.",
        required=True,
    )
    parser.add_argument(
        "--save_path",
        "-s",
        type=str,
        help="Root path to save the preprocessed files (will create subfolder preprocessed)."
        "If not specified, uses the folder with the original data and creates a preprocessed subfolder "
        "(recommended)",
        default=None,
    )
    parser.add_argument(
        "--num_raters",
        "-r",
        type=int,
        help="Number of raters that labeled one image.",
        required=True,
    )
    parser.add_argument(
        "--image_dirs",
        "-i",
        type=str,
        nargs="+",
        help="Name of the directory in root_dir where the original images are located,"
        "e.g. [imagesTr, imagesTs]. If None, assumes [images]",
        default=None,
    )
    parser.add_argument(
        "--label_dirs",
        "-l",
        type=str,
        nargs="+",
        help="Name of the directory in root_dir where the original labels are located,"
        "e.g. [labelsTr, labelsTs]. If None, assumes [labels]. Has to match corresponding image_dirs.",
        default=None,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Which dataset to preprocess, i.e. which dataset is stored in dataset_path."
        "Currently either 'toy' or 'lidc'",
        default=None,
    )
    args = parser.parse_args()
    return args


def preprocess_dataset(
    root_dir: Path,
    save_path: Path,
    num_raters: int,
    image_dirs: List[str] = None,
    label_dirs: List[str] = None,
    dataset: str = None,
    patch_size: int = 64,
    patch_overlap: float = 1,
) -> None:
    """Preprocess the dataset, i.e. normalize the images (z-score normalization) and save

    Args:
        root_dir (Path): The root directory of the dataset
        save_path (Path): Directory where to store the preprocessed dataset
        num_raters (int): Number of raters that labeled one image
        image_dirs (List[str]): Name of the directory in root_dir where the original images are located,
                                e.g. [imagesTr, imagesTs]. If None, assumes [images]
        label_dirs (List[str]): Name of the directory in root_dir where the original labels are located,
                                e.g. [labelsTr, labelsTs]. If None, assumes [labels]. Has to match corresponding image_dirs.
        dataset (str): Name of the dataset to preprocess
        patch_size (int): Size of the patches used for training
        patch_overlap (float): How much the patches overlap
    """
    if image_dirs is None:
        image_dirs = ["images"]
    if label_dirs is None:
        label_dirs = ["labels"]
    for image_dir_name, label_dir_name in zip(image_dirs, label_dirs):
        image_dir = root_dir / image_dir_name
        label_dir = root_dir / label_dir_name
        output_dir_images = save_path / "preprocessed" / image_dir_name
        output_dir_labels = save_path / "preprocessed" / label_dir_name

        os.makedirs(output_dir_images, exist_ok=True)
        os.makedirs(output_dir_labels, exist_ok=True)

        nii_files = subfiles(image_dir, suffix=".nii.gz", join=False)

        all_images = []
        all_labels = []
        for f in nii_files:
            image, _ = load(os.path.join(image_dir, f))
            all_images.append(image)
            raters = []
            for rater in range(num_raters):
                if dataset == "lidc" or dataset == "lidc-idri":
                    label_name = f.replace(
                        ".nii.gz", "_{}_mask.nii.gz".format(str(rater).zfill(2))
                    )
                else:
                    label_name = f.replace(
                        ".nii.gz", "_{}.nii.gz".format(str(rater).zfill(2))
                    )
                if os.path.isfile(
                    os.path.join(
                        label_dir,
                        label_name,
                    )
                ):
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
                image.shape[0] % int(patch_size * patch_overlap)
            )
            pad_size_y = image.shape[1] + (
                image.shape[1] % int(patch_size * patch_overlap)
            )
            pad_size_z = image.shape[2] + (
                image.shape[2] % int(patch_size * patch_overlap)
            )
            image = pad_nd_image(
                image,
                (pad_size_x, pad_size_y, pad_size_z),
                "constant",
                kwargs={"constant_values": image.min()},
            )
            np.save(os.path.join(output_dir_images, f.split(".")[0] + ".npy"), image)
            for rater in range(len(label)):
                label_rater = pad_nd_image(
                    label[rater],
                    (pad_size_x, pad_size_y, pad_size_z),
                    "constant",
                    kwargs={"constant_values": label[rater].min()},
                )

                if dataset == "lidc" or dataset == "lidc-idri":
                    file_name = f"{f.split('.')[0]}_{str(rater).zfill(2)}_mask.npy"
                else:
                    file_name = f"{f.split('.')[0]}_{str(rater).zfill(2)}.npy"
                np.save(output_dir_labels / file_name, label_rater)


def main(args: Namespace):
    dataset_path = Path(args.dataset_path)
    if args.save_path is not None:
        save_path = Path(args.save_path)
    else:
        save_path = dataset_path
    image_dirs = args.image_dirs
    label_dirs = args.label_dirs
    dataset = args.dataset
    num_raters = args.num_raters
    preprocess_dataset(
        root_dir=dataset_path,
        save_path=save_path,
        image_dirs=image_dirs,
        label_dirs=label_dirs,
        dataset=dataset,
        num_raters=num_raters,
    )


if __name__ == "__main__":
    cli_args = main_cli()
    main(args=cli_args)
