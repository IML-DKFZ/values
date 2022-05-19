from __future__ import annotations

import os
from typing import Optional

from msd_datamodule import MSDDataModule, NumpyDataLoader, FixedLengthAugmenter


class HeartDataModule(MSDDataModule):
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
        Pytorch-Lightning DataModule containing the Heart dataset,
        alongside data transformation, loading & preprocessing.

        Args:
            data_input_dir (str): Where to save/load the dataset.
            data_num_folds (int): How many folds to split the dataset into. Creates splits.pkl file if not yet present.
            data_fold_id (int): Which fold to use.
            batch_size (int): Which batch size to use for dataloaders.
            num_workers (int): How many workers to use for loading data. Please set according to your CPU.
        """
        dataset_name = "Task02_Heart"
        super(HeartDataModule, self).__init__(
            dataset_name=dataset_name,
            data_input_dir=data_input_dir,
            data_num_folds=data_num_folds,
            data_fold_id=data_fold_id,
            batch_size=batch_size,
            y_shape=None,
            z_shape=None,
            num_workers=num_workers,
            seed=seed,
            google_drive_id="1wEB2I6S6tQBVEPxir8cA5kFB8gTQadYY",
        )

    @property
    def num_classes(self):
        return 2

    def train_dataloader(self) -> FixedLengthAugmenter:
        """Dataloader for training. Loads numpy data and defines transformations

        Returns:
            [FixedLengthAugmenter]: Multithreaded train dataloader with augmentations
        """
        train_loader = NumpyDataLoader(
            base_dir=os.path.join(
                self.data_input_dir, self.dataset_name, "preprocessed"
            ),
            batch_size=self.batch_size,
            file_pattern="*.npy",
            subject_ids=self.tr_keys,
            slice_offset=80,
        )
        train_augmenter = FixedLengthAugmenter(
            data_loader=train_loader,
            transform=self.get_transforms(mode="train", target_size=128),
            num_processes=self.num_workers,
            num_cached_per_queue=2,
            seeds=[self.seed] * self.num_workers,
            pin_memory=True,
            timeout=10,
            wait_time=0.02,
        )
        return train_augmenter
