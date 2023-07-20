import os.path
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import subfiles

import albumentations as A
import cv2
from tqdm import tqdm

import sys

sys.path.append("../../")
import uncertainty_modeling.data.cityscapes_labels as cs_labels


def main_cli():
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        "-d",
        type=str,
        help="Path to the original dataset.",
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
        "--dataset",
        type=str,
        help="Which dataset to preprocess, i.e. which dataset is stored in dataset_path."
        "Either 'cityscapes' or 'gta'",
        required=True,
    )
    args = parser.parse_args()
    return args


def preprocess_dataset(dataset_dir: Path, save_dir: Path, dataset: str) -> None:
    output_dir_images = save_dir / "preprocessed" / "images"
    output_dir_labels = save_dir / "preprocessed" / "labels"
    # save the color maps of the labels
    output_dir_labels_color = output_dir_labels / "vis"
    output_dir_image_vis = output_dir_images / "vis"

    os.makedirs(output_dir_images, exist_ok=True)
    os.makedirs(output_dir_image_vis, exist_ok=True)
    os.makedirs(output_dir_labels, exist_ok=True)
    os.makedirs(output_dir_labels_color, exist_ok=True)

    if dataset == "cityscapes":
        splits = ["train", "val"]
        images_base_path = dataset_dir / "images" / "leftImg8bit"
        labels_base_path = dataset_dir / "labels" / "gtFine"
        image_dirs = []
        label_dirs = []
        for split in splits:
            splits_image_dir = images_base_path / split
            splits_label_dir = labels_base_path / split
            for city in os.listdir(splits_image_dir):
                image_dir = splits_image_dir / city
                label_dir = splits_label_dir / city
                if image_dir.is_dir():
                    image_dirs.append(image_dir)
                    label_dirs.append(label_dir)
        image_dirs = sorted(image_dirs)
        label_dirs = sorted(label_dirs)
    else:
        image_dirs = [dataset_dir / "images"]
        label_dirs = [dataset_dir / "labels"]

    num_diff_res = 0
    for image_dir, label_dir in zip(image_dirs, label_dirs):
        png_images = subfiles(image_dir, suffix=".png", join=False)
        # avoid mac ._ prefixed images
        png_images = [image for image in png_images if not image.startswith(".")]

        # define transformations
        crop_transform = A.Compose(
            [
                A.CenterCrop(height=1024, width=1912, always_apply=True),
            ]
        )

        for image_name in tqdm(png_images):
            if dataset == "cityscapes":
                image_id = image_name.split("_leftImg8bit")[0]
            else:
                image_id = image_name.split(".")[0]
            image_np_save_path = output_dir_images / f"{image_id}.npy"
            image_png_save_path = output_dir_image_vis / f"{image_id}.png"
            mask_np_save_path = output_dir_labels / f"{image_id}.npy"
            mask_png_save_path = output_dir_labels_color / f"{image_id}.png"

            if (
                image_np_save_path.is_file()
                and image_png_save_path.is_file()
                and mask_np_save_path.is_file()
                and mask_png_save_path.is_file()
            ):
                continue
            if image_name == "15188.png" or image_name == "17705.png":
                continue

            image_path = image_dir / image_name
            if dataset == "cityscapes":
                label_path = label_dir / f"{image_id}_gtFine_labelIds.png"
            else:
                label_path = label_dir / image_name

            image = cv2.imread(str(image_path), -1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask_transform = cv2.imread(str(label_path), -1)

            if image.shape[:2] != mask_transform.shape[:2]:
                num_diff_res += 1
                print(f"Different resolutions between image and mask for {image_name}!")
                continue

            # apply albumentations transforms
            transformed = crop_transform(image=image, mask=mask_transform)
            image = transformed["image"].astype(np.uint8)

            # rescale image and mask to 0.25 of the original size using different interpolation schemes
            image = cv2.resize(
                image, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR
            )
            if dataset == "cityscapes":
                mask_labels = transformed["mask"].astype(np.uint8)
                mask_labels = cv2.resize(
                    mask_labels,
                    (0, 0),
                    fx=0.25,
                    fy=0.25,
                    interpolation=cv2.INTER_NEAREST,
                )

                # ignore labels that are not evaluated
                mask_train_indices = mask_labels.copy()
                for k, v in cs_labels.id2trainId.items():
                    mask_train_indices[mask_labels == k] = v

                # transform the labels to color map for visualization
                mask_color = np.zeros((*mask_train_indices.shape, 3), dtype=np.uint8)
                for k, v in cs_labels.trainId2color.items():
                    mask_color[mask_train_indices == k] = np.array(v)
            else:
                mask_color = transformed["mask"].astype(np.uint8)
                mask_color = cv2.cvtColor(mask_color, cv2.COLOR_BGR2RGB)
                mask_color = cv2.resize(
                    mask_color,
                    (0, 0),
                    fx=0.25,
                    fy=0.25,
                    interpolation=cv2.INTER_NEAREST,
                )

                # transform the color mask to actual labels, use the 19 training labels here
                mask_train_indices = np.apply_along_axis(
                    lambda x: cs_labels.color2trainId.get(tuple(x), 128),
                    axis=-1,
                    arr=mask_color,
                )
                assert (
                    128 not in mask_train_indices
                ), f"Unknown color value in mask for image {image_name}!"

            np.save(image_np_save_path, image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mask_color = cv2.cvtColor(mask_color, cv2.COLOR_RGB2BGR)

            cv2.imwrite(str(image_png_save_path), image)
            np.save(mask_np_save_path, mask_train_indices)
            cv2.imwrite(str(mask_png_save_path), mask_color)


def main(args: Namespace):
    dataset_path = Path(args.dataset_path)
    if args.save_path is not None:
        save_path = Path(args.save_path)
    else:
        save_path = dataset_path
    dataset = args.dataset
    preprocess_dataset(dataset_dir=dataset_path, save_dir=save_path, dataset=dataset)


if __name__ == "__main__":
    cli_args = main_cli()
    main(cli_args)
