import os

import numpy as np

from experiment_version import ExperimentVersion
from medpy.io import load, save


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
        self.pred_prob_dir = (
            self.dataset_path / "pred_prob"
            if os.path.exists(self.dataset_path / "pred_prob")
            else None
        )
        self.image_ids = self._get_image_ids()
        if self.exp_version.pred_model == "Softmax":
            self._setup_pred_entropy_softmax()
        self.unc_path_dict = self._setup_unc_path_dict()

    def get_max_softmax_pred(self, image_id: str):
        probs = []
        for class_prob in range(self.exp_version.n_classes):
            prob_file = os.path.join(
                self.pred_prob_dir,
                f"{image_id}_01_{str(class_prob + 1).zfill(2)}{self.exp_version.unc_ending}",
            )
            image, _ = load(prob_file)
            probs.append(image)
        probs = np.array(probs)
        max_softmax = np.max(probs, axis=0)
        return 1 - max_softmax

    def _setup_pred_entropy_softmax(self):
        if not os.path.exists(self.dataset_path / "pred_entropy"):
            os.makedirs(self.dataset_path / "pred_entropy")
            for image_id in self.image_ids:
                max_softmax = self.get_max_softmax_pred(image_id)
                save(
                    max_softmax,
                    self.dataset_path
                    / "pred_entropy"
                    / f"{image_id}{self.exp_version.unc_ending}",
                )

    def _setup_unc_path_dict(self):
        unc_path_dict = {}
        for unc_type in self.exp_version.unc_types:
            if unc_type == "predictive_uncertainty":
                unc_path_dict[unc_type] = self.dataset_path / "pred_entropy"
            else:
                unc_path_dict[unc_type] = self.dataset_path / unc_type
        return unc_path_dict

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
