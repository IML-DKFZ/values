import importlib
import os
from typing import Optional, Tuple

import torch
import yaml
from argparse import Namespace, ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from msd_datamodule import MSDDataModule
from unet_lightning import UNetExperiment

from pytorch_lightning.callbacks import TQDMProgressBar


def main_cli(
    config_file: str = "./unet_defaults.yml",
    get_nested_dict: bool = False,
) -> Tuple[Namespace, Optional[dict]]:
    """Setting hparams and environment up for the main

    Args:
        config_file (str, optional): path to hyperpameter defaults from light_seg. Defaults to './unet_defaults.yml'.
        get_nested_dict (bool, optional): if True: returns nested subdict to be saved. Defaults to False.

    Returns:
        hparams [Namespace]: all hyperparameters for Trainer, Module, Data and Logger
        nested_dict [dict] : specified hparams which will be saved for easy access
    """
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = UNetExperiment.add_module_specific_args(parser)
    parser = MSDDataModule.add_data_specific_args(parser)
    parser.add_argument(
        "--exp_name", type=str, default="U-Net-Experiment", help="Experiment name."
    )
    parser.add_argument("--version", type=str, default=None, help="Experiment version.")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/home/finja/Documents/SSL-SEG/experiments/debug",
        help="If given, uses this string to create directory to save results in "
        "(be careful, this can overwrite previous results); "
        "otherwise saves logs according to time-stamp",
    )
    parser.add_argument(
        "--datamodule_module_name",
        type=str,
        default="hippocampus_datamodule",
        help="The module name of the datamodule (i.e. the import path)",
    )
    parser.add_argument(
        "--datamodule_class_name",
        type=str,
        default="HippocampusDataModule",
        help="The class name of the datamodule",
    )

    with open(os.path.join(os.path.dirname(__file__), config_file), "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    parser.set_defaults(**config)

    hparams = parser.parse_args()

    # Use Environment Variables if accessible
    if "DATASET_LOCATION" in os.environ.keys():
        hparams.data_input_dir = os.environ["DATASET_LOCATION"]
    if "EXPERIMENT_LOCATION" in os.environ.keys():
        hparams.save_dir = os.environ["EXPERIMENT_LOCATION"]
    if "LSB_JOBID" in os.environ.keys():
        hparams.version = os.environ["LSB_JOBID"]

    if hparams.seed is not None:
        pl.seed_everything(hparams.seed)
        # For 3d Models change determinitic to false
        torch.use_deterministic_algorithms(True, warn_only=True)

    if get_nested_dict is False:
        return hparams, None
    else:
        nested_dict = generate_nested_dict(hparams)
        return hparams, nested_dict


def generate_nested_dict(hparams: Namespace) -> dict:
    """Generates a nested dictionary for easy readability.
    Takes all hparams from Module and DataModule, as well as number of epochs from pl.Trainer

    Args:
        hparams ([Namesapce]): all hyperparameters

    Returns:
        [dict]: nested dictionary with subset of hyperparameters
    """
    parser_module = ArgumentParser()
    parser_module = UNetExperiment.add_module_specific_args(parser_module)
    hparams_module, _ = parser_module.parse_known_args()

    parser_data = ArgumentParser()
    parser_data = MSDDataModule.add_data_specific_args(parser_data)
    hparams_data, _ = parser_data.parse_known_args()

    hparams_train = Namespace()
    hparams_train.max_epochs = hparams.max_epochs

    nested_dict = {
        "train": vars(hparams_train),
        "data": vars(hparams_data),
        "module": vars(hparams_module),
    }
    return nested_dict


def main(hparams: Namespace, nested_dict: Optional[dict] = None):
    """Uses the pl.Trainer to fit & test the model

    Args:
        hparams ([Namespace]): hparams
        nested_dict ([dict], optional): Subset of hparams for saving. Defaults to None.
    """

    logger = TensorBoardLogger(
        save_dir=hparams.save_dir, version=hparams.version, name=hparams.exp_name
    )

    if hparams.progress_bar_refresh_rate is not None:
        progress_bar = TQDMProgressBar(refresh_rate=hparams.progress_bar_refresh_rate)
        delattr(hparams, "progress_bar_refresh_rate")
        trainer = pl.Trainer.from_argparse_args(
            hparams, logger=logger, profiler="simple", callbacks=progress_bar
        )
    else:
        trainer = pl.Trainer.from_argparse_args(
            hparams, logger=logger, profiler="simple"
        )

    DataModule = getattr(
        importlib.import_module(hparams.datamodule_module_name),
        hparams.datamodule_class_name,
    )
    dm = DataModule(**hparams.__dict__)
    hparams.dataset_name = dm.dataset_name
    hparams.num_classes = dm.num_classes

    dm.prepare_data()
    dm.setup("fit")
    model = UNetExperiment(hparams, nested_dict=nested_dict, **vars(hparams))
    trainer.fit(model, datamodule=dm)

    # Testing is currently done in a separate script, use these lines if you want to enable it within the
    # pytorch lightning loop directly after training
    # heart_dm.setup("test")
    # trainer.test(model, datamodule=heart_dm)


if __name__ == "__main__":
    hparams, nested_dict = main_cli(config_file="./unet_defaults.yml")
    main(hparams, nested_dict=nested_dict)
