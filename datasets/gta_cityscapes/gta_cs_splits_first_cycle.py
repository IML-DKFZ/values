import os.path
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
from sklearn.model_selection import KFold


def main_cli():
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        "-d",
        type=str,
        help="Root path to the preprocessed GTA and Cityscapes dataset, i.e. path where the preprocessed "
        "GTA and Cityscapes dataset are located in subfolders OriginalData/preprocessed and "
        "CityScapesOriginalData/preprocessed respectively.",
        required=True,
    )
    parser.add_argument(
        "--original_dataset_path",
        "-o",
        type=str,
        help="Root path to the original GTA and Cityscapes dataset, i.e. path where the original "
        "GTA and Cityscapes dataset are located in subfolders OriginalData and "
        "CityScapesOriginalData respectively. If not given, assumes the same folder as dataset_path",
        default=None,
    )
    parser.add_argument(
        "--splits_path",
        "-s",
        type=str,
        help="Path to store the created splits file. "
        "If not given, the will be stored in the dataset_path under splits/firstCycle/splits.pkl."
        "If given as directory, a file named splits.pkl will be created, otherwise has to be specified as .pkl file",
        default=None,
    )
    args = parser.parse_args()
    return args


def get_cs_splits(base_dir: Path, split: str):
    org_image_dir = (
        base_dir / "CityScapesOriginalData" / "images" / "leftImg8bit" / split
    )
    split_dirs = [
        path
        for path in os.listdir(org_image_dir)
        if os.path.isdir(os.path.join(org_image_dir, path))
    ]
    return sorted(split_dirs)


def create_splits(
    base_dir: Path, orig_base_dir: Path, splits_path: Path, seed, n_splits=5
) -> None:
    np.random.seed(seed)
    gta_image_dir = base_dir / "OriginalData" / "preprocessed" / "images"
    cs_image_dir = base_dir / "CityScapesOriginalData" / "preprocessed" / "images"

    gta_images = [
        (image, "gta")
        for image in os.listdir(gta_image_dir)
        if image.endswith(".npy") and not image.startswith("._")
    ]
    gta_images = sorted(gta_images)
    print(len(gta_images))
    cs_images = [
        (image, "cs")
        for image in os.listdir(cs_image_dir)
        if image.endswith(".npy") and not image.startswith("._")
    ]
    cs_images = sorted(cs_images)
    print(len(cs_images))

    # create cs splits
    # cs train images is ood unlabeled pool for first cycle
    cs_train_cities = get_cs_splits(orig_base_dir, "train")
    cs_train_images = []
    for city in cs_train_cities:
        cs_train_images.extend([image for image in cs_images if city in image[0]])
    # cs val images is ood test
    cs_test_cities = get_cs_splits(orig_base_dir, "val")
    cs_test_images = []
    for city in cs_test_cities:
        cs_test_images.extend([image for image in cs_images if city in image[0]])

    gta_unlabeled_pool_indices = np.random.choice(
        len(gta_images), size=len(cs_train_images), replace=False
    )
    # gta unlabeled pool is id unlabeled pool
    gta_unlabeled_pool_images = [
        image
        for index, image in enumerate(gta_images)
        if index in gta_unlabeled_pool_indices
    ]
    gta_train_test_images = [
        image
        for index, image in enumerate(gta_images)
        if index not in gta_unlabeled_pool_indices
    ]

    num_test = int(0.25 * len(gta_train_test_images))
    gta_test_indices = np.random.choice(
        len(gta_train_test_images), size=num_test, replace=False
    )
    # id test images
    gta_test_images = [
        image
        for index, image in enumerate(gta_train_test_images)
        if index in gta_test_indices
    ]
    # train / val images
    gta_train_val_images = [
        image
        for index, image in enumerate(gta_train_test_images)
        if index not in gta_test_indices
    ]

    splits = []
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    # create fold dictionary and append it to splits
    for i, (train_idx, val_idx) in enumerate(kfold.split(gta_train_val_images)):
        gta_train_images = [
            image
            for index, image in enumerate(gta_train_val_images)
            if index in train_idx
        ]
        gta_val_images = [
            image
            for index, image in enumerate(gta_train_val_images)
            if index in val_idx
        ]
        split_dict = dict()
        split_dict["train"] = gta_train_images
        split_dict["val"] = gta_val_images
        split_dict["id_test"] = gta_test_images
        split_dict["ood_test"] = cs_test_images
        split_dict["id_unlabeled_pool"] = gta_unlabeled_pool_images
        split_dict["ood_unlabeled_pool"] = cs_train_images
        splits.append(split_dict)

    with open(splits_path, "wb") as f:
        pickle.dump(splits, f)


def main(args: Namespace):
    pl.seed_everything(123)
    dataset_path = Path(args.dataset_path)
    if args.original_dataset_path is None:
        original_dataset_path = dataset_path
    else:
        original_dataset_path = Path(args.original_dataset_path)
    if args.splits_path is not None:
        splits_path = Path(args.splits_path)
        if not str(splits_path).endswith(".pkl"):
            splits_path = splits_path / "splits.pkl"
    else:
        splits_path = dataset_path / "splits" / "firstCycle" / "splits.pkl"
    os.makedirs(splits_path.parent, exist_ok=True)
    create_splits(
        base_dir=dataset_path,
        orig_base_dir=original_dataset_path,
        splits_path=splits_path,
        seed=123,
    )


if __name__ == "__main__":
    cli_args = main_cli()
    main(args=cli_args)
