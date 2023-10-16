import json
import os

import numpy as np
from sklearn.calibration import _sigmoid_calibration as calib
from sklearn import utils as sk_utils
from sklearn import preprocessing as sk_preprocess
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
                ignore_mask = reference_segs != ignore_value
                a, b = calib(-unc_map[ignore_mask], rater_correct[ignore_mask])
            else:
                a, b = calib(-unc_map.flatten(), np.array(rater_correct).flatten())
            ps_params_dict[unc_type]["a"].append(a)
            ps_params_dict[unc_type]["b"].append(b)
        ps_params_dict[unc_type]["a"] = np.mean(np.array(ps_params_dict[unc_type]["a"]))
        ps_params_dict[unc_type]["b"] = np.mean(np.array(ps_params_dict[unc_type]["b"]))
    with open(
        val_exp_dataloader.exp_version.exp_path / "platt_scale_params.json", "w"
    ) as f:
        json.dump(ps_params_dict, f, indent=2)


def platt_scale_confid(uncalib_confid, platt_scale_file, uncertainty):
    with open(platt_scale_file) as f:
        params_dict = json.load(f)
    params = params_dict[uncertainty]
    return 1 / (1 + np.exp(uncalib_confid * params["a"] + params["b"]))


def calib_stats(correct, calib_confids):
    # calib_confids = np.clip(self.confids, 0, 1)
    n_bins = 20
    y_true = sk_utils.column_or_1d(correct)
    y_prob = sk_utils.column_or_1d(calib_confids)

    if y_prob.min() < 0 or y_prob.max() > 1:
        raise ValueError(
            "y_prob has values outside [0, 1] and normalize is " "set to False."
        )

    labels = np.unique(y_true)
    if len(labels) > 2:
        raise ValueError(
            "Only binary classification is supported. " f"Provided labels {labels}."
        )
    y_true = sk_preprocess.label_binarize(y_true, classes=labels)[:, 0]

    bins = np.linspace(0.0, 1.0 + 1e-8, n_bins + 1)

    binids = np.digitize(y_prob, bins) - 1

    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    nonzero = bin_total != 0
    num_nonzero = len(nonzero[nonzero == True])
    prob_true = bin_true[nonzero] / bin_total[nonzero]
    prob_pred = bin_sums[nonzero] / bin_total[nonzero]
    prob_total = bin_total[nonzero] / bin_total.sum()

    bin_discrepancies = np.abs(prob_true - prob_pred)
    return bin_discrepancies, prob_total, num_nonzero


def calc_ace(correct, calib_confids):
    bin_discrepancies, _, num_nonzero = calib_stats(correct, calib_confids)
    return (1 / num_nonzero) * np.sum(bin_discrepancies)


def calibration_error(exp_dataloader: ExperimentDataloader, ignore_value=None):
    calib_dict = {}
    calib_dict["mean"] = {}
    for unc_type in exp_dataloader.exp_version.unc_types:
        aces_unc = []
        for image_id in tqdm(exp_dataloader.image_ids):
            if image_id not in calib_dict.keys():
                calib_dict[image_id] = {}
            reference_segs = exp_dataloader.get_reference_segs(image_id)
            pred_seg = exp_dataloader.get_mean_pred_seg(image_id)
            unc_map = exp_dataloader.get_unc_map(image_id, unc_type)
            # 2d unc map is loaded in shape (W, H)
            if pred_seg.shape != unc_map.shape:
                unc_map = np.swapaxes(unc_map, 0, 1)
            pred_seg = np.repeat(pred_seg[np.newaxis, :], reference_segs.shape[0], 0)
            unc_map = np.repeat(unc_map[np.newaxis, :], reference_segs.shape[0], 0)
            rater_correct = (reference_segs == pred_seg).astype(int)
            platt_scale_file = (
                exp_dataloader.exp_version.exp_path / "platt_scale_params.json"
            )
            if ignore_value is not None:
                ignore_mask = reference_segs != ignore_value
                unc_map = platt_scale_confid(
                    -unc_map[ignore_mask],
                    platt_scale_file=platt_scale_file,
                    uncertainty=unc_type,
                )
                ace = calc_ace(rater_correct[ignore_mask], unc_map)
                calib_dict[image_id][unc_type] = {"metrics": {"ace": ace}}
                aces_unc.append(ace)
            else:
                unc_map = platt_scale_confid(
                    -unc_map.flatten(),
                    platt_scale_file=platt_scale_file,
                    uncertainty=unc_type,
                )
                ace = calc_ace(rater_correct.flatten(), unc_map)
                calib_dict[image_id][unc_type] = {"metrics": {"ace": ace}}
                aces_unc.append(ace)
        calib_dict["mean"][unc_type] = {"metrics": {"ace": np.mean(np.array(aces_unc))}}
    save_path = exp_dataloader.dataset_path / "calibration.json"
    with open(save_path, "w") as f:
        json.dump(calib_dict, f, indent=2)


def main(exp_dataloader: ExperimentDataloader, ignore_value=None):
    platt_scale_params_file = (
        exp_dataloader.exp_version.exp_path / "platt_scale_params.json"
    )
    # replace by checking whether platt scale params file exists
    if not os.path.isfile(platt_scale_params_file):
        val_exp_dataloader = ExperimentDataloader(exp_dataloader.exp_version, "val")
        platt_scale_params(val_exp_dataloader, ignore_value=ignore_value)
    calibration_error(exp_dataloader, ignore_value=ignore_value)
