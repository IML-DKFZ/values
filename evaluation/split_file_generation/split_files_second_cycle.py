import json
import os
from pathlib import Path

import numpy as np

from evaluation.experiment_dataloader import ExperimentDataloader
import pickle as pkl

from evaluation.utils.sort_uncertainties import sort_uncertainties


def get_splits_first_cycle(base_split_path, shift: str = None):
    if shift is not None:
        save_path = base_split_path / shift / "firstCycle" / "splits.pkl"
    else:
        save_path = base_split_path / "firstCycle" / "splits.pkl"
    with open(save_path, "rb") as f:
        splits = pkl.load(f)
    return splits


def get_aggregated_uncertainties(aggregated_unc_path):
    with open(aggregated_unc_path) as f:
        aggregated_uncertainties = json.load(f)
    return aggregated_uncertainties


def get_samples_to_query(sorted_uncertainties, acquisition_size: float):
    num_to_query = int(len(sorted_uncertainties) * acquisition_size)
    return sorted_uncertainties[:num_to_query]


def update_splits(splits, samples_to_query, unc_file_ending):
    samples_to_query = [
        sample.replace(unc_file_ending, ".npy") for sample in samples_to_query
    ]
    print(len(samples_to_query))
    num_unlabeled_before = len(splits[0]["id_unlabeled_pool"]) + len(
        splits[0]["ood_unlabeled_pool"]
    )
    num_train_before = len(splits[0]["train"])
    is_tuple = False
    if type(splits[0]["train"][0]) == tuple:
        samples_to_query = [
            (sample, "gta") if sample[0].isdigit() else (sample, "cs")
            for sample in samples_to_query
        ]
        is_tuple = True
    for sample in samples_to_query:
        if sample in splits[0]["id_unlabeled_pool"]:
            if not is_tuple:
                sample_index = np.argwhere(splits[0]["id_unlabeled_pool"] == sample)
            else:
                sample_compare = sample[0]
                split_compare = np.array([s[0] for s in splits[0]["id_unlabeled_pool"]])
                sample_index = np.argwhere(split_compare == sample_compare)
            if sample_index.size > 1:
                print(f"Sample {sample} found multiple times")
                continue
            else:
                splits[0]["id_unlabeled_pool"] = np.delete(
                    splits[0]["id_unlabeled_pool"], sample_index[0][0], axis=0
                )
                if not is_tuple:
                    splits[0]["train"] = np.append(splits[0]["train"], sample)
                else:
                    splits[0]["train"] = np.append(splits[0]["train"], [sample], axis=0)
        elif sample in splits[0]["ood_unlabeled_pool"]:
            if not is_tuple:
                sample_index = np.argwhere(splits[0]["ood_unlabeled_pool"] == sample)
            else:
                sample_compare = sample[0]
                split_compare = np.array(
                    [s[0] for s in splits[0]["ood_unlabeled_pool"]]
                )
                sample_index = np.argwhere(split_compare == sample_compare)
            if sample_index.size > 1:
                print(f"Sample {sample} found multiple times")
                continue
            else:
                splits[0]["ood_unlabeled_pool"] = np.delete(
                    splits[0]["ood_unlabeled_pool"], sample_index[0][0], axis=0
                )
                if not is_tuple:
                    splits[0]["train"] = np.append(splits[0]["train"], sample)
                else:
                    splits[0]["train"] = np.append(splits[0]["train"], [sample], axis=0)
        else:
            print("Could not find sample {}!".format(sample))
    num_unlabeled_after = len(splits[0]["id_unlabeled_pool"]) + len(
        splits[0]["ood_unlabeled_pool"]
    )
    num_train_after = len(splits[0]["train"])
    print(num_unlabeled_before, num_unlabeled_after)
    print(num_train_before, num_train_after)
    print("======")
    assert num_unlabeled_after == num_unlabeled_before - len(samples_to_query)
    assert num_train_after == num_train_before + len(samples_to_query)
    return splits


def save_splits(
    new_splits, base_split_path, shift, pred_model, uncertainty, aggregation, seed
):
    if shift is not None:
        save_dir = (
            base_split_path
            / shift
            / "secondCycle"
            / pred_model
            / uncertainty
            / aggregation
        )
    else:
        save_dir = (
            base_split_path / "secondCycle" / pred_model / uncertainty / aggregation
        )
    os.makedirs(save_dir, exist_ok=True)
    save_path = save_dir / f"splits_seed{seed}.pkl"
    with open(save_path, "wb") as f:
        pkl.dump(new_splits, f)


def generate_split_file(
    exp_dataloader: ExperimentDataloader,
    base_splits_path,
):
    base_splits_path = Path(base_splits_path)
    if "shift" in exp_dataloader.exp_version.version_params:
        shift = exp_dataloader.exp_version.version_params["shift"]
    else:
        shift = None
    for (
        unc,
        aggregated_unc_path,
    ) in exp_dataloader.get_aggregated_unc_files_dict().items():
        for aggregation in exp_dataloader.exp_version.aggregations:
            splits = get_splits_first_cycle(base_splits_path, shift=shift)
            uncertainties = get_aggregated_uncertainties(aggregated_unc_path)
            sorted_uncertainties = sort_uncertainties(uncertainties, aggregation)
            samples_to_query = get_samples_to_query(sorted_uncertainties, 0.5)
            new_splits = update_splits(
                splits,
                samples_to_query,
                unc_file_ending=exp_dataloader.exp_version.unc_ending,
            )
            save_splits(
                new_splits=new_splits,
                base_split_path=base_splits_path,
                shift=shift,
                pred_model=exp_dataloader.exp_version.pred_model,
                uncertainty=unc,
                aggregation=aggregation,
                seed=exp_dataloader.exp_version.version_params["seed"],
            )
