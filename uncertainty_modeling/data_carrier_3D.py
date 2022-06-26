import json
import os
from typing import Dict

import numpy as np
import torch
from medpy.io import load, save


class DataCarrier3D:
    def __init__(self):
        self.data = {}
        self.save_dir = None
        self.save_input_dir = None
        self.save_gt_dir = None
        self.save_pred_dir = None
        self.save_pred_prob_dir = None

    def _create_save_dirs(self, root_dir: str, exp_name: str, version: int) -> None:
        """
        Create the directories to store the test results in.
        Args:
            root_dir: The root directory where the experiments are saved
            exp_name: Name of the experiment
            version: version of the experiment
        """
        self.save_dir = os.path.join(root_dir, exp_name, "test_results", str(version))
        self.save_input_dir = os.path.join(self.save_dir, "input")
        self.save_gt_dir = os.path.join(self.save_dir, "gt_seg")
        self.save_pred_dir = os.path.join(self.save_dir, "pred_seg")
        self.save_pred_prob_dir = os.path.join(self.save_dir, "pred_prob")

        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.save_input_dir, exist_ok=True)
        os.makedirs(self.save_gt_dir, exist_ok=True)
        os.makedirs(self.save_pred_dir, exist_ok=True)
        os.makedirs(self.save_pred_prob_dir, exist_ok=True)

    @staticmethod
    def load_image(sample) -> Dict:
        """
        Get an input dictionary from a single 2D slice of a 3D image
        Args:
            sample [Dict]: Dictionary containing the path to the 3D image and label
                           and the index of the slice which should be loaded.
                           Keys are 'image_path', 'label_path'
        Returns:
            input [Dict]: Dict with the necessary keys for inference including the loaded data& label slice.
                          Keys are 'image_paths', 'label_paths', 'data', 'seg'
        """
        input = {}
        input["image_paths"] = [sample["image_path"]]
        input["label_paths"] = [sample["label_paths"]]
        input["crop_idx"] = [sample["crop_idx"]]

        image_array = np.load(sample["image_path"], mmap_mode="r")
        input["org_image_size"] = [image_array.shape]
        image_patch = image_array[
            sample["crop_idx"][0][0] : sample["crop_idx"][0][1],
            sample["crop_idx"][1][0] : sample["crop_idx"][1][1],
            sample["crop_idx"][2][0] : sample["crop_idx"][2][1],
        ]
        input["data"] = np.expand_dims(image_patch, 0)

        if sample["label_paths"] is not None:
            label_patches = []
            for label_path in sample["label_paths"]:
                label_array = np.load(label_path, mmap_mode="r")
                label_patch = label_array[
                    sample["crop_idx"][0][0] : sample["crop_idx"][0][1],
                    sample["crop_idx"][1][0] : sample["crop_idx"][1][1],
                    sample["crop_idx"][2][0] : sample["crop_idx"][2][1],
                ]
                label_patches.append(label_patch)
            label_patches = np.array(label_patches)
            input["seg"] = np.expand_dims(label_patches, 1)
        return input

    def concat_data(
        self,
        batch: Dict,
        softmax_pred: torch.Tensor,
        n_pred: int = 1,
        pred_idx: int = 0,
    ) -> None:
        """
        Concatenate the data
        Stores the data in self.data which is a dictionary containing the image paths as keys
        and the other information like the image and the segmentation data.
        Args:
            batch: batch to insert in self.data
            softmax_pred: softmax prediction to insert in self.data
        """
        for index, image_path in enumerate(batch["image_paths"]):
            if image_path not in self.data:
                self.data[image_path] = {}
                self.data[image_path]["label_paths"] = batch["label_paths"][index]
                self.data[image_path]["softmax_pred"] = np.zeros(
                    (n_pred, 2, *batch["org_image_size"][index])
                )
                self.data[image_path]["num_predictions"] = np.zeros(
                    (2, *batch["org_image_size"][index])
                )

                self.data[image_path]["data"] = np.zeros(batch["org_image_size"][index])

                self.data[image_path]["seg"] = np.zeros(
                    [len(batch["label_paths"][index]), *batch["org_image_size"][index]]
                )

            crop_idx = batch["crop_idx"][index]
            self.data[image_path]["data"][
                crop_idx[0][0] : crop_idx[0][1],
                crop_idx[1][0] : crop_idx[1][1],
                crop_idx[2][0] : crop_idx[2][1],
            ] += (
                batch["data"][index].cpu().detach().numpy().squeeze()
            )
            self.data[image_path]["seg"][
                crop_idx[0][0] : crop_idx[0][1],
                crop_idx[1][0] : crop_idx[1][1],
                crop_idx[2][0] : crop_idx[2][1],
            ] += (
                batch["seg"][:, index, :, :, :].cpu().detach().numpy().squeeze()
            )
            self.data[image_path]["softmax_pred"][
                pred_idx,
                :,
                crop_idx[0][0] : crop_idx[0][1],
                crop_idx[1][0] : crop_idx[1][1],
                crop_idx[2][0] : crop_idx[2][1],
            ] += (
                softmax_pred[index].cpu().detach().numpy()
            )
            self.data[image_path]["num_predictions"][
                :,
                crop_idx[0][0] : crop_idx[0][1],
                crop_idx[1][0] : crop_idx[1][1],
                crop_idx[2][0] : crop_idx[2][1],
            ] += 1

    def save_data(
        self,
        root_dir: str,
        exp_name: str,
        version: int,
        org_data_path: str = None,
    ) -> None:
        """
        Saves the data according to the folder structure in _create_save_dirs
        Args:
            root_dir: The root directory where the experiments are saved
            exp_name: Name of the experiment
            version: version of the experiment
            org_data_path: The path to the original data to infer header information
        """
        self._create_save_dirs(root_dir=root_dir, exp_name=exp_name, version=version)
        for key, value in self.data.items():
            data = np.asarray(
                value["data"] / np.clip(value["num_predictions"], 1, None)[0]
            )
            gt_seg = np.asarray(
                value["seg"] / np.clip(value["num_predictions"], 1, None)[0]
            )
            softmax_pred = value["softmax_pred"] / np.clip(
                value["num_predictions"], 1, None
            )

            if org_data_path:
                _, header = load(
                    os.path.join(
                        org_data_path, key.split("/")[-1].split(".")[0] + ".nii.gz"
                    )
                )

            else:
                header = False
            save(
                data,
                os.path.join(
                    self.save_input_dir,
                    key.split("/")[-1].split(".")[0] + ".nii.gz",
                ),
                header,
            )

            for seg_idx in range(gt_seg.shape[0]):
                save(
                    gt_seg[seg_idx],
                    os.path.join(
                        self.save_gt_dir,
                        "{}_{}.nii.gz".format(
                            key.split("/")[-1].split(".")[0], str(seg_idx).zfill(2)
                        ),
                    ),
                    header,
                )
            for pred_idx in range(softmax_pred.shape[0]):
                pred_seg = np.argmax(softmax_pred[pred_idx], axis=0)
                pred_seg = np.asarray(pred_seg)
                save(
                    pred_seg.astype(np.uint8),
                    os.path.join(
                        self.save_pred_dir,
                        "{}_{}.nii.gz".format(
                            key.split("/")[-1].split(".")[0], str(pred_idx + 1).zfill(2)
                        ),
                    ),
                    header,
                )
                for class_idx in range(softmax_pred.shape[1]):
                    class_prob = softmax_pred[pred_idx, class_idx, :, :, :]
                    save(
                        class_prob,
                        os.path.join(
                            self.save_pred_prob_dir,
                            "{}_{}_{}.nii.gz".format(
                                key.split("/")[-1].split(".")[0],
                                str(pred_idx + 1).zfill(2),
                                str(class_idx + 1).zfill(2),
                            ),
                        ),
                        header,
                    )

            if "pred_entropy" in value:
                save_pred_entropy_dir = os.path.join(self.save_dir, "pred_entropy")
                os.makedirs(save_pred_entropy_dir, exist_ok=True)
                pred_entropy = np.asarray(
                    value["pred_entropy"]
                    / np.clip(value["num_predictions"], 1, None)[0]
                )
                save(
                    pred_entropy,
                    os.path.join(
                        save_pred_entropy_dir,
                        key.split("/")[-1].split(".")[0] + ".nii.gz",
                    ),
                    header,
                )
            if "aleatoric_uncertainty" in value:
                save_aleatoric_uncertainty_dir = os.path.join(
                    self.save_dir, "aleatoric_uncertainty"
                )
                os.makedirs(save_aleatoric_uncertainty_dir, exist_ok=True)
                aleatoric_uncertainty = np.asarray(
                    value["aleatoric_uncertainty"]
                    / np.clip(value["num_predictions"], 1, None)[0]
                )
                save(
                    aleatoric_uncertainty,
                    os.path.join(
                        save_aleatoric_uncertainty_dir,
                        key.split("/")[-1].split(".")[0] + ".nii.gz",
                    ),
                    header,
                )
            if "epistemic_uncertainty" in value:
                save_epistemic_uncertainty_dir = os.path.join(
                    self.save_dir, "epistemic_uncertainty"
                )
                os.makedirs(save_epistemic_uncertainty_dir, exist_ok=True)
                epistemic_uncertainty = np.asarray(
                    value["epistemic_uncertainty"]
                    / np.clip(value["num_predictions"], 1, None)[0]
                )
                save(
                    epistemic_uncertainty,
                    os.path.join(
                        save_epistemic_uncertainty_dir,
                        key.split("/")[-1].split(".")[0] + ".nii.gz",
                    ),
                    header,
                )

    def log_metrics(self) -> None:
        """
        Save the metrics to a json file
        """
        filename = os.path.join(self.save_dir, "metrics.json")
        metrics_dict = {}
        mean_dict = {}
        for image_path, value in self.data.items():
            metrics_dict[image_path] = {}
            for metric, score in value["metrics"].items():
                metrics_dict[image_path][metric] = score
                if metric not in mean_dict:
                    mean_dict[metric] = []
                mean_dict[metric].append(score)
        metrics_dict["mean"] = {}
        for metric, scores in mean_dict.items():
            metrics_dict["mean"][metric] = np.asarray(scores).mean()
        with open(filename, "w") as f:
            json.dump(metrics_dict, f, indent=2)
