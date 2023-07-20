import os
import pickle
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
from sklearn.model_selection import KFold

sys.path.append("../../")
from utils.subfiles import subfiles


def main_cli():
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        "-d",
        type=str,
        help="Path to the toy dataset. If not given, the arguments --splits_path, --train_dir and --test_dir "
        "have to be specified",
        default=None,
    )
    parser.add_argument(
        "--splits_path",
        "-s",
        type=str,
        help="Path to store the created splits file. If not given, the argument --dataset_path has to be specified."
        "If given as directory, a file named splits.pkl will be created, otherwise has to be specified as .pkl file",
        default=None,
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        help="Path to the train images, used to create train / val split. "
        "If not given, the argument --dataset_path has to be specified.",
        default=None,
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        help="Path to the test images. "
        "If not given, the argument --dataset_path has to be specified.",
        default=None,
    )
    args = parser.parse_args()
    return args


def create_splits(splits_path, train_dir, test_dir, seed, n_splits=5) -> None:
    """Saves a pickle file containing the splits for k-fold cv on the dataset

    Args:
        splits_path: The output directory where to save the splits file
        train_dir: The directory of the preprocessed training/ validation images
        test_dir: The directory of the preprocessed test images
        seed: The seed for the splits
        n_splits: Number of folds
    """
    np.random.seed(seed)
    train_npy_files = subfiles(str(train_dir), suffix=".npy", join=False, sort=True)
    test_npy_files = subfiles(str(test_dir), suffix=".npy", join=False, sort=True)

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

    with open(splits_path, "wb") as f:
        pickle.dump(splits, f)


def main(args: Namespace):
    seed = 123
    pl.seed_everything(seed)
    if args.dataset_path is None:
        if args.splits_path is None:
            print(
                "If you didn't specify the dataset path, "
                "you need to specify where to store the splits file!"
            )
            return
        if args.train_dir is None:
            print(
                "If you didn't specify the dataset path, "
                "you need to specify where the training images are stored!"
            )
            return
        if args.test_dir is None:
            print(
                "If you didn't specify the dataset path, "
                "you need to specify where the test images are stored!"
            )
            return
    if args.dataset_path is not None:
        dataset_path = Path(args.dataset_path)
        train_dir = dataset_path / "preprocessed" / "imagesTr"
        test_dir = dataset_path / "preprocessed" / "imagesTs"
        splits_path = dataset_path / "splits.pkl"
    if args.train_dir is not None:
        train_dir = Path(args.train_dir)
    if args.test_dir is not None:
        test_dir = Path(args.test_dir)
    if args.splits_path is not None:
        splits_path = Path(args.splits_path)
        if not str(splits_path).endswith(".pkl"):
            splits_path = splits_path / "splits.pkl"
    os.makedirs(splits_path.parent, exist_ok=True)

    create_splits(
        splits_path=splits_path, train_dir=train_dir, test_dir=test_dir, seed=123
    )
    return


if __name__ == "__main__":
    cli_args = main_cli()
    main(args=cli_args)
