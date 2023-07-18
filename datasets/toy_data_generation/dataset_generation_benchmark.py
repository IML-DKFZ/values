import os.path
import shutil
import subprocess
from argparse import ArgumentParser, Namespace, RawTextHelpFormatter
from pathlib import Path


def main_cli():
    parser = ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        "--base_save_path",
        type=str,
        help="Path to the folder the images and labels will be stored. Will create a folder with the case name"
        "(e.g. Case_1) and the subfolders imagesTr, labelsTr, imagesTs, labelsTs in there.",
        required=True,
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Name of the dataset that you want to generate. One of the folder names in the config folder.\n"
        "Case_1: Training models on data with induced AU; testing on i.i.d. data also containing AU.\n"
        "Case_2: Training models on data without ambiguity; testing on i.i.d. data and shifted data.\n"
        "Case_3a: Training models on data with and without blur/ambiguity; "
        "testing on i.i.d. data and shifted data without blur.\n"
        "Case_3b: Training models on data with and without blur/ambiguity; "
        "testing on i.i.d. data and shifted data without blur and i.i.d data with blur.",
        default="Case_1",
    )
    args = parser.parse_args()
    return args


def make_datasets(config_files, base_save_path, train: bool):
    folder_ending = "Tr" if train else "Ts"
    images_save_dir = base_save_path / f"images{folder_ending}"
    labels_save_dir = base_save_path / f"labels{folder_ending}"

    for config_file in config_files:
        subprocess.call(
            [
                "python",
                "dataset_generation.py",
                "--json_config",
                config_file,
                "--save_path",
                images_save_dir,
            ]
        )
        labels_created_dir = images_save_dir / "segmentation"
        shutil.copytree(labels_created_dir, labels_save_dir, dirs_exist_ok=True)
        shutil.rmtree(labels_created_dir)


def main(args: Namespace):
    dataset_name = args.dataset_name
    if not os.path.isdir(f"configs/{dataset_name}"):
        print(
            "The dataset name that you specified does not have config files. "
            "Provide a dataset name that in listed in the config directory!"
        )
        return
    base_save_path = Path(args.base_save_path) / dataset_name
    os.makedirs(base_save_path, exist_ok=True)
    configs_train = [
        f"configs/{dataset_name}/{config_file}"
        for config_file in os.listdir(f"configs/{dataset_name}")
        if config_file
        if config_file.startswith("train")
    ]
    configs_test = [
        f"configs/{dataset_name}/{config_file}"
        for config_file in os.listdir(f"configs/{dataset_name}")
        if config_file
        if config_file.startswith("test")
    ]
    make_datasets(config_files=configs_train, base_save_path=base_save_path, train=True)
    make_datasets(config_files=configs_test, base_save_path=base_save_path, train=False)


if __name__ == "__main__":
    cli_args = main_cli()
    main(cli_args)
