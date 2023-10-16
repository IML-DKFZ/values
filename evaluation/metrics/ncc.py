import json

import numpy as np

from evaluation.experiment_dataloader import ExperimentDataloader
from tqdm import tqdm


def compute_ncc(gt_unc_map: np.array, pred_unc_map: np.array):
    """
    Compute the normalized cross correlation between a ground truth uncertainty and a predicted uncertainty map,
    to determine how similar the maps are.
    :param gt_unc_map: the ground truth uncertainty map based on the rater variability
    :param pred_unc_map: the predicted uncertainty map
    :return: float: the normalized cross correlation between gt and predicted uncertainty map
    """
    mu_gt = np.mean(gt_unc_map)
    mu_pred = np.mean(pred_unc_map)
    sigma_gt = np.std(gt_unc_map, ddof=1)
    sigma_pred = np.std(pred_unc_map, ddof=1)
    gt_norm = gt_unc_map - mu_gt
    pred_norm = pred_unc_map - mu_pred
    prod = np.sum(np.multiply(gt_norm, pred_norm))
    ncc = (1 / (np.size(gt_unc_map) * sigma_gt * sigma_pred)) * prod
    return ncc


def main(exp_dataloader: ExperimentDataloader):
    ncc_dict = {}
    ncc_dict["mean"] = {}
    for unc_type in exp_dataloader.exp_version.unc_types:
        nccs_unc = []
        for image_id in tqdm(exp_dataloader.image_ids):
            if image_id not in ncc_dict.keys():
                ncc_dict[image_id] = {}
            gt_unc_map = exp_dataloader.get_gt_unc_map(image_id)
            pred_unc_map = exp_dataloader.get_unc_map(image_id, unc_type)
            ncc = compute_ncc(gt_unc_map, pred_unc_map)
            ncc_dict[image_id][unc_type] = {"metrics": {"ncc": ncc}}
            nccs_unc.append(ncc)
        ncc_dict["mean"][unc_type] = {"metrics": {"ncc": np.mean(np.array(nccs_unc))}}
    save_path = exp_dataloader.dataset_path / "ambiguity_modeling.json"
    with open(save_path, "w") as f:
        json.dump(ncc_dict, f, indent=2)
