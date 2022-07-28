import os

import hydra.utils
import torch
from argparse import Namespace, ArgumentParser

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf, open_dict
from uncertainty_modeling.lightning_experiment import LightningExperiment


def pl_cli():
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    hparams, _ = parser.parse_known_args()
    return hparams


@hydra.main(config_path="configs", config_name="default_config")
def main(cfg_hydra: DictConfig):
    """Uses the pl.Trainer to fit & test the model

    Args:
        hparams ([Namespace]): hparams
        nested_dict ([dict], optional): Subset of hparams for saving. Defaults to None.
    """
    config = pl_cli()
    config = OmegaConf.create(vars(config))
    with open_dict(cfg_hydra):
        config.merge_with(cfg_hydra)
    config = cfg_hydra
    # Use Environment Variables if accessible
    if "DATASET_LOCATION" in os.environ.keys():
        config.data_input_dir = os.environ["DATASET_LOCATION"]
    if "EXPERIMENT_LOCATION" in os.environ.keys():
        config.save_dir = os.environ["EXPERIMENT_LOCATION"]
    if "LSB_JOBID" in os.environ.keys() and config.version is None:
        config.version = os.environ["LSB_JOBID"]

    if config.seed is not None:
        pl.seed_everything(config.seed)
        torch.use_deterministic_algorithms(True, warn_only=True)

    logger = hydra.utils.instantiate(config.logger, version=config.version)
    progress_bar = hydra.utils.instantiate(config.progress_bar)
    trainer = pl.Trainer.from_argparse_args(
        Namespace(**config),
        logger=logger,
        profiler="simple",
        callbacks=progress_bar,
    )

    dm = hydra.utils.instantiate(
        config.datamodule, data_input_dir=config.data_input_dir
    )
    dm.prepare_data()
    dm.setup("fit")
    model = LightningExperiment(config, **config)
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
