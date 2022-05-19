from __future__ import annotations

import os
from typing import Optional, List
import random
import pickle
import tarfile
import fnmatch
import numpy as np
import pytorch_lightning as pl
from sklearn.model_selection import KFold
from argparse import ArgumentParser

from google_drive_downloader import GoogleDriveDownloader as Gdd

from medpy.io import load
from batchgenerators.augmentations.utils import pad_nd_image
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.spatial_transforms import (
    MirrorTransform,
    ResizeTransform,
    SpatialTransform,
)
from batchgenerators.transforms.utility_transforms import NumpyToTensor
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter

from msd_datamodule import MSDDataModule


class HippocampusDataModule(MSDDataModule):
    def __init__(
        self,
        data_input_dir: Optional[str] = None,
        data_num_folds: int = 5,
        data_fold_id: int = 0,
        batch_size: int = 16,
        num_workers: int = 8,
        seed: int = 42,
        *args,
        **kwargs,
    ):
        """
        Pytorch-Lightning DataModule containing the Hippcampus dataset,
        alongside data transformation, loading & preprocessing.

        Args:
            data_input_dir (str): Where to save/load the dataset.
            data_num_folds (int): How many folds to split the dataset into. Creates splits.pkl file if not yet present.
            data_fold_id (int): Which fold to use.
            batch_size (int): Which batch size to use for dataloaders.
            num_workers (int): How many workers to use for loading data. Please set according to your CPU.
        """
        dataset_name = "Task04_Hippocampus"
        super(HippocampusDataModule, self).__init__(
            dataset_name=dataset_name,
            data_input_dir=data_input_dir,
            data_num_folds=data_num_folds,
            data_fold_id=data_fold_id,
            batch_size=batch_size,
            y_shape=64,
            z_shape=64,
            num_workers=num_workers,
            seed=seed,
            google_drive_id="1RzPB1_bqzQhlWvU-YGvZzhx2omcDh38C",
        )

    @property
    def num_classes(self):
        return 3


class FixedLengthAugmenter(MultiThreadedAugmenter):
    """Subclass of batchgenerators' MultiThreadedAugmenter to enable multithreaded dataloading"""

    def __len__(self):
        return len(self.generator)
