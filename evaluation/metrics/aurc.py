"""
------------------------------------------------------------------------------
Code adapted for segmentation and mainly from:
https://github.com/IML-DKFZ/fd-shifts/blob/main/fd_shifts/analysis/metrics.py
------------------------------------------------------------------------------
"""

import json
import numpy as np

from evaluation.experiment_dataloader import ExperimentDataloader


def rc_curve_stats(
    risks: np.array, confids: np.array
) -> tuple[list[float], list[float], list[float]]:
    coverages = []
    selective_risks = []
    assert (
        len(risks.shape) == 1 and len(confids.shape) == 1 and len(risks) == len(confids)
    )

    n_samples = len(risks)
    idx_sorted = np.argsort(confids)

    coverage = n_samples
    error_sum = sum(risks[idx_sorted])

    coverages.append(coverage / n_samples)
    selective_risks.append(error_sum / n_samples)

    weights = []

    tmp_weight = 0
    for i in range(0, len(idx_sorted) - 1):
        coverage = coverage - 1
        error_sum = error_sum - risks[idx_sorted[i]]
        tmp_weight += 1
        if i == 0 or confids[idx_sorted[i]] != confids[idx_sorted[i - 1]]:
            coverages.append(coverage / n_samples)
            selective_risks.append(error_sum / (n_samples - 1 - i))
            weights.append(tmp_weight / n_samples)
            tmp_weight = 0

    # add a well-defined final point to the RC-curve.
    if tmp_weight > 0:
        coverages.append(0)
        selective_risks.append(selective_risks[-1])
        weights.append(tmp_weight / n_samples)

    return coverages, selective_risks, weights


def aurc(risks: np.array, confids: np.array):
    _, risks, weights = rc_curve_stats(risks, confids)
    return sum(
        [(risks[i] + risks[i + 1]) * 0.5 * weights[i] for i in range(len(weights))]
    )


def eaurc(risks: np.array, confids: np.array):
    """Compute normalized AURC, i.e. subtract AURC of optimal CSF (given fixed risks)."""
    n = len(risks)
    # optimal confidence sorts risk. Asencding here because we start from coverage 1/n
    selective_risks = np.sort(risks).cumsum() / np.arange(1, n + 1)
    aurc_opt = selective_risks.sum() / n
    return aurc(risks, confids) - aurc_opt


def get_risk(image_id: str, metrics_file: str):
    with open(metrics_file, "r") as f:
        metrics = json.load(f)
    if image_id not in metrics.keys():
        key = [k for k in metrics.keys() if k.split("/")[-1].split(".")[0] == image_id]
        if len(key) > 1:
            print(
                f"Found multiple matches for image id {image_id}. Using the first match {key[0]}"
            )
        if "dice" not in metrics[key[0]].keys():
            return 1 - metrics[key[0]]["metrics"]["dice"]
        return 1 - metrics[key[0]]["dice"]
    if "dice" not in metrics[image_id].keys():
        return 1 - metrics[image_id]["metrics"]["dice"]
    return 1 - metrics[image_id]["dice"]


def get_dice(image_id: str, metrics_file: str):
    with open(metrics_file, "r") as f:
        metrics = json.load(f)
    if image_id not in metrics.keys():
        key = [k for k in metrics.keys() if k.split("/")[-1].split(".")[0] == image_id]
        if len(key) > 1:
            print(
                f"Found multiple matches for image id {image_id}. Using the first match {key[0]}"
            )
        if "dice" not in metrics[key[0]].keys():
            return metrics[key[0]]["metrics"]["dice"]
        return metrics[key[0]]["dice"]
    if "dice" not in metrics[image_id].keys():
        return metrics[image_id]["metrics"]["dice"]
    return metrics[image_id]["dice"]


def get_confid(
    image_name: str, aggregated_unc_file: str, aggregation_level: str, unc_file_ending
):
    with open(aggregated_unc_file, "r") as f:
        unc = json.load(f)
    unc_image_name = f"{image_name}{unc_file_ending}"
    return -unc[unc_image_name][aggregation_level]["max_score"]


def get_risks_and_confids(
    dataset_path, image_ids, unc_type, aggregation, unc_file_ending
):
    risks = []
    confids = []
    dices = []
    for image in image_ids:
        risk = get_risk(image, dataset_path / "metrics.json")
        risks.append(risk)
        dice = get_dice(image, dataset_path / "metrics.json")
        dices.append(dice)
        unc_file = dataset_path / f"aggregated_{unc_type}.json"
        confid = get_confid(image, unc_file, aggregation, unc_file_ending)
        confids.append(confid)
    return risks, confids, dices


def main(exp_dataloader: ExperimentDataloader):
    aggregations = exp_dataloader.exp_version.aggregations
    unc_types = exp_dataloader.exp_version.unc_types
    results_dict = {"mean": {}}
    for unc_type in unc_types:
        results_dict["mean"][unc_type] = {}
        for aggregation in aggregations:
            results_dict["mean"][unc_type][aggregation] = {}
            results_dict["mean"][unc_type][aggregation]["metrics"] = {}
            risks, confids, _ = get_risks_and_confids(
                dataset_path=exp_dataloader.dataset_path,
                image_ids=exp_dataloader.image_ids,
                unc_type=unc_type,
                aggregation=aggregation,
                unc_file_ending=exp_dataloader.exp_version.unc_ending,
            )
            aurc_score = aurc(np.array(risks), np.array(confids))
            eaurc_score = eaurc(np.array(risks), np.array(confids))
            results_dict["mean"][unc_type][aggregation]["metrics"]["aurc"] = aurc_score
            results_dict["mean"][unc_type][aggregation]["metrics"][
                "eaurc"
            ] = eaurc_score
    with open(exp_dataloader.dataset_path / "failure_detection.json", "w") as f:
        json.dump(results_dict, f, indent=2)
