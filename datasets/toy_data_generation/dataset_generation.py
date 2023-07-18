import argparse
import json
import os
import random
from argparse import ArgumentParser

import numpy as np
from scipy.ndimage import gaussian_filter

from stl_to_nifty import (
    meshes_to_numpy,
    embed_object_in_image,
    embed_object_in_image_negative_offset,
    add_noise,
    numpy_to_nifti,
)


def main_cli():
    parser = ArgumentParser()

    parser.add_argument(
        "--json_config",
        type=str,
        help="Load settings from file in json format. Command line options override values in file.",
        default=None,
    )

    parser.add_argument(
        "--input_files",
        type=str,
        nargs="+",
        help="The input stl files that should be parsed to nifti files",
    )

    parser.add_argument(
        "--save_path",
        type=str,
        help="Path to the folder where the nifti files should be stored",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        help="Number of samples to create for the dataset",
        default=10,
    )

    parser.add_argument(
        "--image_size",
        type=int,
        nargs="+",
        help="Size of the resulting nifti image. Can be one int (then x=y=z) or three ints for the three dimensions",
        default=[256],
    )

    parser.add_argument(
        "--min_object_ratio",
        type=int,
        help="Minimum ratio that the object size should be compared to the image size",
        default=10,
    )

    parser.add_argument(
        "--max_object_ratio",
        type=int,
        help="Maximum ratio that the object size should be compared to the image size",
        default=2,
    )

    parser.add_argument(
        "--gauss_sigma",
        type=int,
        help="The sigma for gaussian blur if blur is enabled",
        default=8,
    )

    parser.add_argument(
        "--object_gray",
        help="Do not render the object with value 1 in the image but with a gray value between 0.5 and 0.9 instead",
        action="store_true",
    )

    parser.add_argument("--blur", help="Apply gaussian blur", action="store_true")

    parser.add_argument("--noise", help="Add noise in background", action="store_true")

    parser.add_argument(
        "--segmentation",
        help="Also create segmentation for dataset",
        action="store_true",
    )

    parser.add_argument(
        "--all_raters_same",
        help="Whether all raters should provide the same segmentation mask",
        action="store_true",
    )

    parser.add_argument(
        "--n_raters",
        type=int,
        help="The number of raters if segmentation is enabled",
        default=1,
    )

    parser.add_argument(
        "--object_over_border",
        help="Place the object such that it is partially outside the image border",
        action="store_true",
    )

    parser.add_argument(
        "--sample_offset",
        type=int,
        help="Offset for naming of files (If you want to continue a dataset in a folder with different configuration)",
        default=0,
    )

    parser.add_argument(
        "--seed", type=int, help="Seed for random number generation", default=22
    )

    args = parser.parse_args()

    if args.json_config:
        with open(args.json_config, "rt") as f:
            t_args = argparse.Namespace()
            t_args.__dict__.update(json.load(f))
            args = parser.parse_args(namespace=t_args)

    if len(args.image_size) == 1:
        args.image_size = (args.image_size[0], args.image_size[0], args.image_size[0])
    elif len(args.image_size) == 3:
        args.image_size = tuple(args.image_size)
    else:
        raise argparse.ArgumentError(
            args.image_size, "Image size has to be exaclty 1 or 3 values"
        )

    return args


def create_segmentations(n_raters, image, sample_idx, all_rater_same: bool = False):
    os.makedirs(os.path.join(args.save_path, "segmentation"), exist_ok=True)
    if n_raters == 1:
        thresholds = [0.1]
    elif all_rater_same:
        thresholds = [0.1] * n_raters
    else:
        perc_range = 1 - 0.1
        perc_step = perc_range / (n_raters - 1)
        perc_thresholds = np.arange(0.1, 1 + perc_step, perc_step)
        all_object_pixels = np.count_nonzero(image >= 0.1)
        object_ratio = all_object_pixels / image.size
        perc_thresholds *= object_ratio
        thresholds = np.quantile(image, (1 - perc_thresholds))
    for rater_idx, threshold in enumerate(thresholds):
        seg = np.where(image >= threshold, 1, 0)
        save_file_name = os.path.join(
            args.save_path,
            "segmentation",
            "{}_{}.nii.gz".format(str(sample_idx).zfill(4), str(rater_idx).zfill(2)),
        )
        seg = seg.astype(np.intc)
        numpy_to_nifti(seg, save_file_name)


def args_to_json(args):
    suffix = 1
    filename = os.path.join(args.save_path, f"dataset_info_{suffix}.json")
    while os.path.isfile(filename):
        suffix += 1
        filename = os.path.join(
            args.save_path, "dataset_info_{}.json".format(str(suffix))
        )
    with open(filename, "w") as f:
        json.dump(vars(args), f, indent=2)


def create_dataset_samples(args):
    os.makedirs(args.save_path, exist_ok=True)
    for sample_idx in range(args.n_samples):
        object_numpy = meshes_to_numpy(
            args.input_files,
            random.randint(
                int(max(args.image_size) / args.min_object_ratio),
                int(max(args.image_size) / args.max_object_ratio),
            ),
        )
        max_offset = [
            args.image_size[0] - object_numpy.shape[0],
            args.image_size[1] - object_numpy.shape[1],
            args.image_size[2] - object_numpy.shape[2],
        ]
        if not args.object_over_border:
            offset = [
                random.randint(0, max_offset[0]),
                random.randint(0, max_offset[1]),
                random.randint(0, max_offset[2]),
            ]
            image = embed_object_in_image(offset, object_numpy, args.image_size)
        else:
            min_offset = [
                int(-2 * object_numpy.shape[0] / 3),
                int(-2 * object_numpy.shape[1] / 3),
                int(-2 * object_numpy.shape[2] / 3),
            ]
            rand_number = random.randint(1, 7)
            binary_number = format(rand_number, "b").zfill(3)
            offset_0 = (
                random.randint(min_offset[0], 0)
                if int(binary_number[0])
                else random.randint(0, max_offset[0])
            )
            offset_1 = (
                random.randint(min_offset[1], 0)
                if int(binary_number[1])
                else random.randint(0, max_offset[1])
            )
            offset_2 = (
                random.randint(min_offset[2], 0)
                if int(binary_number[2])
                else random.randint(0, max_offset[2])
            )
            offset = [offset_0, offset_1, offset_2]
            image = embed_object_in_image_negative_offset(
                offset, object_numpy, args.image_size
            )
            if random.random() > 0.5:
                image = np.fliplr(image)
            if random.random() > 0.5:
                image = np.flipud(image)
        if args.object_gray:
            image *= random.uniform(0.5, 0.9)
        if args.blur:
            image = gaussian_filter(image, sigma=args.gauss_sigma)

        if args.segmentation:
            create_segmentations(
                args.n_raters,
                image,
                args.sample_offset + sample_idx,
                all_rater_same=args.all_raters_same,
            )

        if args.noise:
            image = add_noise(0.5, image)
        save_file_name = os.path.join(
            args.save_path,
            "{}.nii.gz".format(str(args.sample_offset + sample_idx).zfill(4)),
        )
        numpy_to_nifti(image, save_file_name)
    args_to_json(args)


if __name__ == "__main__":
    args = main_cli()
    random.seed(args.seed)
    np.random.seed(args.seed)
    create_dataset_samples(args)
