import fnmatch
import json
from bisect import bisect
import os
from typing import Optional, List, Dict

import numpy as np
import torch
from medpy.io import load, save


class DataCarrier:
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

    def get_data_samples(
        self,
        base_dir: str,
        pattern: str = "*.npy",
        slice_offset: int = 5,
        subject_ids: Optional[List[str]] = None,
        slice_axis=0,
    ) -> List[dict]:
        """
        Return a list of all possible input samples in the dataset by returning all possible slices for each subject id.

        Args:
            base_dir (str): Directory where preprocessed numpy files reside. Should contain subfolders imagesTr and labelsTr
            pattern (str): Pattern to match os.walk filenames against.
            slice_offset (int): Offset possible slices in the dataset.
            subject_ids (list/array): Which subject IDs to load.
                                      If None, all files with matching pattern will be included

        Returns:
            samples [List[dict]]: All possible slices for each subject id.
        """
        samples = []

        (image_dir, _, image_filenames) = next(
            os.walk(os.path.join(base_dir, "imagesTr"))
        )
        (label_dir, _, label_filenames) = next(
            os.walk(os.path.join(base_dir, "labelsTr"))
        )
        for image_filename in sorted(fnmatch.filter(image_filenames, pattern)):
            if (
                subject_ids is not None and image_filename in subject_ids
            ) or subject_ids is None:
                image_path = os.path.join(image_dir, image_filename)
                image_array = np.load(image_path, mmap_mode="r")
                file_len = image_array.shape[slice_axis]

                label_path = os.path.join(label_dir, image_filename)

                samples.extend(
                    [
                        {
                            "image_path": image_path,
                            "label_path": label_path,
                            "slice_idx": i,
                        }
                        for i in range(slice_offset, file_len - slice_offset)
                    ]
                )
        return samples

    def load_slice(self, sample) -> Dict:
        """
        Get an input dictionary from a single 2D slice of a 3D image
        Args:
            sample [Dict]: Dictionary containing the path to the 3D image and label
                           and the index of the slice which should be loaded.
                           Keys are 'image_path', 'label_path', 'slice_idx'
        Returns:
            input [Dict]: Dict with the necessary keys for inference including the loaded data& label slice.
                          Keys are 'image_paths', 'label_paths', 'slice_idxs', 'data', 'seg'
        """
        input = {}
        input["image_paths"] = [sample["image_path"]]
        input["label_paths"] = [sample["label_path"]]
        input["slice_idxs"] = [sample["slice_idx"]]

        image_array = np.load(sample["image_path"], mmap_mode="r")
        image_slice = np.expand_dims(image_array[sample["slice_idx"]], axis=0)
        data = [image_slice]
        input["data"] = np.asarray(data)

        if sample["label_path"] is not None:
            label_array = np.load(sample["label_path"], mmap_mode="r")
            slice_label = np.expand_dims(label_array[sample["slice_idx"]], axis=0)
            input["seg"] = np.asarray([slice_label])
        return input

    def concat_data(self, batch: Dict, softmax_pred: torch.Tensor):
        """
        Concatenate the data, i.e. sort the slice into the right position of the respective image.
        Stores the data in self.data which is a dictionary containing the image paths as keys
        and the other information like which slices are already added for the image and the slice data (sorted).
        Args:
            batch: batch to insert in self.data
            softmax_pred: softmax prediction to insert in self.data
        """
        for index, image_path in enumerate(batch["image_paths"]):
            if image_path not in self.data:
                self.data[image_path] = {}
                self.data[image_path]["label_path"] = batch["label_paths"][index]
                self.data[image_path]["slice_idxs"] = []
                self.data[image_path]["data"] = []
                self.data[image_path]["seg"] = []
                self.data[image_path]["softmax_pred"] = []
                self.data[image_path]["pred_seg"] = []
            insert_indx = bisect(
                self.data[image_path]["slice_idxs"], batch["slice_idxs"][index]
            )
            self.data[image_path]["slice_idxs"].insert(
                insert_indx, batch["slice_idxs"][index]
            )
            self.data[image_path]["data"].insert(
                insert_indx, batch["data"][index].cpu().detach().numpy().squeeze()
            )
            self.data[image_path]["seg"].insert(
                insert_indx, batch["seg"][index].cpu().detach().numpy().squeeze()
            )
            self.data[image_path]["softmax_pred"].insert(
                insert_indx, softmax_pred[index].cpu().detach().numpy()
            )
            predicted_segmentation = torch.argmax(softmax_pred[index], dim=0)
            self.data[image_path]["pred_seg"].insert(
                insert_indx, predicted_segmentation.cpu().detach().numpy()
            )

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
        """
        self._create_save_dirs(root_dir=root_dir, exp_name=exp_name, version=version)
        for key, value in self.data.items():
            data = np.asarray(value["data"])
            gt_seg = np.asarray(value["seg"])
            softmax_pred = np.asarray(value["softmax_pred"])
            pred_seg = np.asarray(value["pred_seg"])

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

            save(
                gt_seg,
                os.path.join(
                    self.save_gt_dir, key.split("/")[-1].split(".")[0] + ".nii.gz"
                ),
                header,
            )

            for class_idx in range(softmax_pred.shape[1]):
                class_prob = softmax_pred[:, class_idx, :, :].squeeze()
                save(
                    class_prob,
                    os.path.join(
                        self.save_pred_prob_dir,
                        "{}_{}.nii.gz".format(
                            key.split("/")[-1].split(".")[0],
                            str(class_idx + 1).zfill(4),
                        ),
                    ),
                    header,
                )

            save(
                pred_seg.astype(np.uint8),
                os.path.join(
                    self.save_pred_dir, key.split("/")[-1].split(".")[0] + ".nii.gz"
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
