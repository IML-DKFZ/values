import os

import hydra
import numpy as np

from evaluation.utils.set_seed import set_seed
from experiment_version import ExperimentVersion
from medpy.io import load, save


class ExperimentDataloader:
    def __init__(self, exp_version: ExperimentVersion, dataset_split):
        self.exp_version = exp_version
        set_seed(int(self.exp_version.version_params["seed"]))
        self.dataset_split = dataset_split
        self.dataset_path = (
            exp_version.exp_path / self.dataset_split
            if self.dataset_split
            else exp_version.exp_path
        )
        self.pred_seg_dir = self.dataset_path / "pred_seg"
        self.pred_prob_dir = (
            self.dataset_path / "pred_prob"
            if os.path.exists(self.dataset_path / "pred_prob")
            else None
        )
        self.image_ids = sorted(self._get_image_ids())
        if self.exp_version.pred_model == "Softmax":
            self._setup_pred_entropy_softmax()
        self.unc_path_dict = self._setup_unc_path_dict()
        if self.exp_version.datamodule_config is not None:
            self.dataloader = self.setup_dataloader()
            self.ref_seg_dir = None
        else:
            self.dataloader = None
            self.ref_seg_dir = self.dataset_path / "gt_seg"

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
            for image_name in os.listdir(self.pred_seg_dir)
            if image_name.endswith(self.exp_version.image_ending)
        )

    def get_pred_seg_paths(self, image_id):
        return [
            self.pred_seg_dir / image_path
            for image_path in os.listdir(self.pred_seg_dir)
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

    def get_aggregated_unc_files_dict(self):
        aggregated_unc_file_dict = {}
        for unc in self.unc_path_dict.keys():
            if os.path.isfile(self.dataset_path / f"aggregated_{unc}.json"):
                aggregated_unc_file_dict[unc] = (
                    self.dataset_path / f"aggregated_{unc}.json"
                )
        return aggregated_unc_file_dict

    def setup_dataloader(self):
        dm = hydra.utils.instantiate(
            self.exp_version.datamodule_config,
            test_split=self.dataset_split,
            _recursive_=False,
        )
        dm.setup("test")
        return dm.test_dataloader()

    def get_reference_segs(self, image_id):
        if self.dataloader is not None:
            idx = self.dataloader.dataset.image_ids.index(image_id)
            data = self.dataloader.dataset.__getitem__(idx)
            return data["seg"].squeeze().numpy()
        else:
            n_reference_segs = self.exp_version.n_reference_segs
            reference_segs_paths = [
                self.ref_seg_dir / f"{image_id}_{i:02d}{self.exp_version.image_ending}"
                for i in range(n_reference_segs)
            ]
            reference_segs = []
            for reference_seg_path in reference_segs_paths:
                reference_seg, _ = load(reference_seg_path)
                reference_segs.append(reference_seg)
            return np.array(reference_segs)

    def get_gt_unc_map(self, image_id):
        if self.exp_version.gt_unc_map_loading is None:
            n_reference_segs = self.exp_version.n_reference_segs
            reference_segs_paths = [
                self.ref_seg_dir / f"{image_id}_{i:02d}{self.exp_version.image_ending}"
                for i in range(n_reference_segs)
            ]
            reference_segs = []
            for reference_seg_path in reference_segs_paths:
                reference_seg, _ = load(reference_seg_path)
                reference_segs.append(reference_seg)
            reference_segs = np.array(reference_segs)
            per_pixel_variance = np.var(reference_segs, axis=0)
        else:
            per_pixel_variance = hydra.utils.instantiate(
                self.exp_version.gt_unc_map_loading,
                image_id=image_id,
                dataloader=self.dataloader,
            )
        return per_pixel_variance

    def get_mean_pred_seg(self, image_id):
        pred_seg_path = (
            self.pred_seg_dir
            / f"{image_id}_{'mean' if self.exp_version.pred_model != 'Softmax' else '01'}{self.exp_version.image_ending}"
        )
        if self.exp_version.pred_seg_loading is None:
            pred_seg, _ = load(pred_seg_path)
        else:
            pred_seg = hydra.utils.instantiate(
                self.exp_version.pred_seg_loading, pred_seg_path=pred_seg_path
            )
        return pred_seg

    def get_unc_map(self, image_id, unc_type):
        unc_map_path = (
            self.unc_path_dict[unc_type] / f"{image_id}{self.exp_version.unc_ending}"
        )
        unc_map, _ = load(unc_map_path)
        return unc_map
