import os

from experiment_version import ExperimentVersion
from medpy.io import load


class ExperimentDataloader:
    def __init__(self, exp_version: ExperimentVersion, dataset_split):
        self.exp_version = exp_version
        self.dataset_split = dataset_split
        self.dataset_path = (
            exp_version.exp_path / self.dataset_split
            if self.dataset_split
            else exp_version.exp_path
        )
        self.pred_seg_path = self.dataset_path / "pred_seg"
        self.image_ids = self._get_image_ids()

    def _get_image_ids(self):
        return set(
            "_".join(image_name.split("_")[:-1])
            for image_name in os.listdir(self.pred_seg_path)
            if image_name.endswith(self.exp_version.image_ending)
        )

    def get_pred_seg_paths(self, image_id):
        return [
            self.pred_seg_path / image_path
            for image_path in os.listdir(self.pred_seg_path)
            if image_path.startswith(image_id)
            and image_path.endswith(self.exp_version.image_ending)
        ]

    def get_pred_segs(self, image_id):
        image_paths = self.get_pred_seg_paths(image_id)
        pred_segs = []
        for image_path in image_paths:
            image, _ = load(image_path)
            pred_segs.append(image)
        return pred_segs
