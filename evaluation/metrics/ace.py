import json
import os
from pathlib import Path

import numpy as np
from sklearn.calibration import _sigmoid_calibration as calib
from tqdm import tqdm

from evaluation.experiment_dataloader import ExperimentDataloader


def platt_scale_params(val_exp_dataloader: ExperimentDataloader, ignore_value=None):
    ps_params_dict = {}
    for unc_type in val_exp_dataloader.exp_version.unc_types:
        ps_params_dict[unc_type] = {"a": [], "b": []}
        for image_id in tqdm(val_exp_dataloader.image_ids):
            reference_segs = val_exp_dataloader.get_reference_segs(image_id)
            pred_seg = val_exp_dataloader.get_mean_pred_seg(image_id)
            unc_map = val_exp_dataloader.get_unc_map(image_id, unc_type)
            # 2d unc map is loaded in shape (W, H)
            if pred_seg.shape != unc_map.shape:
                unc_map = np.swapaxes(unc_map, 0, 1)
            pred_seg = np.repeat(pred_seg[np.newaxis, :], reference_segs.shape[0], 0)
            unc_map = np.repeat(unc_map[np.newaxis, :], reference_segs.shape[0], 0)
            rater_correct = (reference_segs == pred_seg).astype(int)
            if ignore_value is not None:
                ignore_mask = reference_segs != 255
                a, b = calib(-unc_map[ignore_mask], rater_correct[ignore_mask])
            else:
                a, b = calib(-unc_map.flatten(), np.array(rater_correct).flatten())
            ps_params_dict[unc_type]["a"].append(a)
            ps_params_dict[unc_type]["b"].append(b)
        ps_params_dict[unc_type]["a"] = np.mean(np.array(ps_params_dict[unc_type]["a"]))
        ps_params_dict[unc_type]["b"] = np.mean(np.array(ps_params_dict[unc_type]["b"]))
    save_path_ref = str(val_exp_dataloader.exp_version.exp_path).replace(
        str(val_exp_dataloader.exp_version.base_path),
        "/home/kckahl/Refactoring/LIDC-IDRI/ActiveLearning/FirstCycle",
    )
    save_path_ref = Path(save_path_ref)
    os.makedirs(save_path_ref, exist_ok=True)
    print(save_path_ref)
    with open(save_path_ref / "platt_scale_params.json", "w") as f:
        json.dump(ps_params_dict, f, indent=2)


def main(exp_dataloader: ExperimentDataloader):
    save_path_ref = str(exp_dataloader.exp_version.exp_path).replace(
        str(exp_dataloader.exp_version.base_path),
        "/home/kckahl/Refactoring/LIDC-IDRI/ActiveLearning/FirstCycle",
    )
    save_path_ref = Path(save_path_ref)
    platt_scale_params_file = save_path_ref / "platt_scale_params.json"
    # replace by checking whether platt scale params file exists
    if not os.path.isfile(platt_scale_params_file):
        val_exp_dataloader = ExperimentDataloader(exp_dataloader.exp_version, "val")
        platt_scale_params(val_exp_dataloader)
