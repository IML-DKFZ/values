import json
from pathlib import Path

import jsbeautifier
import numpy as np
from sklearn.metrics import roc_curve, auc

from evaluation.experiment_dataloader import ExperimentDataloader
from evaluation.split_file_generation.split_files_second_cycle import (
    get_splits_first_cycle,
    get_aggregated_uncertainties,
    get_samples_to_query,
)
from evaluation.utils.sort_uncertainties import sort_uncertainties


def is_ood_toy(sample):
    # In used toy datasets, samples smaller than 21 are OoD
    # Caution: This is currently hardcoded
    if int(sample.split(".")[0]) > 20:
        return False
    else:
        return True


def is_ood_split(sample, splits, fold=0):
    id_unlabeled_pool = splits[fold]["id_unlabeled_pool"]
    if type(id_unlabeled_pool[0]) == tuple:
        id_unlabeled_pool = [image[0] for image in id_unlabeled_pool]
    ood_unlabeled_pool = splits[fold]["ood_unlabeled_pool"]
    if type(ood_unlabeled_pool[0]) == tuple:
        ood_unlabeled_pool = [image[0] for image in ood_unlabeled_pool]
    if sample in id_unlabeled_pool:
        sample_index = np.argwhere(id_unlabeled_pool == sample)
        if sample_index.size > 1:
            print("Sample found multiple times")
        else:
            return False
    elif sample in ood_unlabeled_pool:
        sample_index = np.argwhere(ood_unlabeled_pool == sample)
        if sample_index.size > 1:
            print("Sample found multiple times")
        else:
            return True
    else:
        print("Could not find sample {}!".format(sample))
    return None


def is_ood(sample, splits, fold=0):
    if splits is None:
        return is_ood_toy(sample)
    else:
        return is_ood_split(sample, splits, fold)


def get_ood_detection_rate(samples_to_query, splits=None, fold=0):
    samples_to_query = [f"{sample.split('.')[0]}.npy" for sample in samples_to_query]
    id = 0
    ood = 0
    for sample in samples_to_query:
        if not is_ood(sample=sample, splits=splits, fold=fold):
            id += 1
        elif is_ood(sample=sample, splits=splits, fold=fold):
            ood += 1
        else:
            print(f"Error for sample {sample}!")
    if splits is None:
        # In toy dataset, there are 21 OoD samples.
        # Caution: This is currently hardcoded
        num_ood_samples = 21
    else:
        num_ood_samples = len(splits[fold]["ood_unlabeled_pool"])
    ood_detection_rate = ood / num_ood_samples
    print("OOD Detection rate: ", ood_detection_rate)
    return ood_detection_rate


def get_auroc_input(uncertainties, aggregation, splits=None, fold=0):
    y_labels = []
    unc_scores = []
    for sample, unc in uncertainties.items():
        sample = f"{sample.split('.')[0]}.npy"
        if not is_ood(sample=sample, splits=splits, fold=fold):
            y_labels.append(0)
            unc_scores.append(unc[aggregation]["max_score"])
        elif is_ood(sample=sample, splits=splits, fold=fold):
            y_labels.append(1)
            unc_scores.append(unc[aggregation]["max_score"])
        else:
            print("Error for sample {}!".format(sample))
    return y_labels, unc_scores


def ood_detection(
    exp_dataloader: ExperimentDataloader,
    base_splits_path=None,
):
    base_splits_path = Path(base_splits_path) if base_splits_path is not None else None
    if "shift" in exp_dataloader.exp_version.version_params:
        shift = exp_dataloader.exp_version.version_params["shift"]
    else:
        shift = None
    ood_det_dict = {"mean": {}}
    for (
        unc,
        aggregated_unc_path,
    ) in exp_dataloader.get_aggregated_unc_files_dict().items():
        ood_det_dict["mean"][unc] = {}
        for aggregation in exp_dataloader.exp_version.aggregations:
            if base_splits_path is not None:
                splits = get_splits_first_cycle(base_splits_path, shift=shift)
            else:
                splits = None
            uncertainties = get_aggregated_uncertainties(aggregated_unc_path)
            sorted_uncertainties = sort_uncertainties(uncertainties, aggregation)
            samples_to_query = get_samples_to_query(sorted_uncertainties, 0.5)
            ood_detection_rate = get_ood_detection_rate(
                samples_to_query=samples_to_query,
                splits=splits,
                fold=exp_dataloader.exp_version.version_params["fold"],
            )
            y_true, y_score = get_auroc_input(
                uncertainties=uncertainties,
                aggregation=aggregation,
                splits=splits,
                fold=exp_dataloader.exp_version.version_params["fold"],
            )
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            ood_det_dict["mean"][unc][aggregation] = {
                "metrics": {"ood_detection_rate": ood_detection_rate, "auroc": roc_auc}
            }
            print("AUROC: ", roc_auc)
            save_path = exp_dataloader.exp_version.exp_path / "ood_detection.json"
            opts = jsbeautifier.default_options()
            opts.indent_size = 4
            with open(save_path, "w") as f:
                f.write(jsbeautifier.beautify(json.dumps(ood_det_dict), opts))
