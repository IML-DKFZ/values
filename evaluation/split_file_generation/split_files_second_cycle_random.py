import os
from pathlib import Path
import pickle as pkl

import numpy as np

from evaluation.experiment_dataloader import ExperimentDataloader
from evaluation.split_file_generation.split_files_second_cycle import (
    get_splits_first_cycle,
)


def get_samples_to_query_random(splits, acquisition_size: float):
    all_unlabeled = np.concatenate(
        (splits[0]["id_unlabeled_pool"], splits[0]["ood_unlabeled_pool"])
    )
    num_to_query = int(len(all_unlabeled) * acquisition_size)
    if all_unlabeled.ndim > 1:
        indices = np.random.choice(all_unlabeled.shape[0], num_to_query, replace=False)
        samples_to_query = all_unlabeled[indices]
        return [tuple(row) for row in samples_to_query.tolist()]
    return np.random.choice(all_unlabeled, num_to_query, replace=False).tolist()


def get_samples_to_query_random_best(splits, acquisition_size: float):
    all_unlabeled = np.concatenate(
        (splits[0]["id_unlabeled_pool"], splits[0]["ood_unlabeled_pool"])
    )
    all_ood = splits[0]["ood_unlabeled_pool"]
    all_id = splits[0]["id_unlabeled_pool"]
    num_to_query = int(len(all_unlabeled) * acquisition_size)
    num_to_query_id = num_to_query - len(all_ood)
    selected_id = np.random.choice(all_id, num_to_query_id, replace=False)
    return np.concatenate((all_ood, selected_id)).tolist()


def get_samples_to_query_random_worst(splits, acquisition_size: float):
    all_unlabeled = np.concatenate(
        (splits[0]["id_unlabeled_pool"], splits[0]["ood_unlabeled_pool"])
    )
    all_id = splits[0]["id_unlabeled_pool"]
    num_to_query = int(len(all_unlabeled) * acquisition_size)
    return np.random.choice(all_id, num_to_query, replace=False).tolist()


def get_samples_to_query(splits, random_type: str, acquisition_size: float):
    if random_type == "random":
        return get_samples_to_query_random(splits, acquisition_size)
    elif random_type == "best":
        return get_samples_to_query_random_best(splits, acquisition_size)
    elif random_type == "worst":
        return get_samples_to_query_random_worst(splits, acquisition_size)


def update_splits(splits, samples_to_query, random_type):
    if type(samples_to_query[0]) != tuple:
        samples_to_query = [
            sample.replace(".nii.gz", ".npy") for sample in samples_to_query
        ]
        is_tuple = False
    else:
        is_tuple = True
    print(len(samples_to_query))
    num_unlabeled_before = len(splits[0]["id_unlabeled_pool"]) + len(
        splits[0]["ood_unlabeled_pool"]
    )
    num_train_before = len(splits[0]["train"])
    for sample in samples_to_query:
        if sample in splits[0]["id_unlabeled_pool"]:
            if not is_tuple:
                sample_index = np.argwhere(splits[0]["id_unlabeled_pool"] == sample)
            else:
                # mask = np.array(list(map(lambda x: x == sample, splits[0]["id_unlabeled_pool"])))
                sample_compare = sample[0]
                split_compare = np.array([s[0] for s in splits[0]["id_unlabeled_pool"]])
                sample_index = np.argwhere(split_compare == sample_compare)
            if sample_index.size > 1:
                print("Sample found multiple times")
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
            if random_type == "worst":
                assert "OOD sample in random worst!"
            if not is_tuple:
                sample_index = np.argwhere(splits[0]["ood_unlabeled_pool"] == sample)
            else:
                # mask = np.array(list(map(lambda x: x == sample, splits[0]["id_unlabeled_pool"])))
                sample_compare = sample[0]
                split_compare = np.array(
                    [s[0] for s in splits[0]["ood_unlabeled_pool"]]
                )
                sample_index = np.argwhere(split_compare == sample_compare)
            if sample_index.size > 1:
                print("Sample found multiple times")
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


def save_splits(new_splits, base_split_path, shift, pred_model, random_type, seed):
    if shift is not None:
        save_dir = (
            base_split_path
            / shift
            / "secondCycle"
            / pred_model
            / "random"
            / random_type
        )
    else:
        save_dir = base_split_path / "secondCycle" / pred_model / "random" / random_type
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "splits_seed{}.pkl".format(seed))
    with open(save_path, "wb") as f:
        pkl.dump(new_splits, f)


def generate_split_file(
    exp_dataloader: ExperimentDataloader, base_splits_path, random_types
):
    base_splits_path = Path(base_splits_path)
    if "shift" in exp_dataloader.exp_version.version_params:
        shift = exp_dataloader.exp_version.version_params["shift"]
    else:
        shift = None
    for random_type in random_types:
        np.random.seed(int(exp_dataloader.exp_version.version_params["seed"]))
        splits = get_splits_first_cycle(base_splits_path, shift=shift)
        samples_to_query = get_samples_to_query(
            splits=splits, random_type=random_type, acquisition_size=0.5
        )
        new_splits = update_splits(
            splits=splits,
            samples_to_query=samples_to_query,
            random_type=random_type,
        )
        save_splits(
            new_splits=new_splits,
            base_split_path=base_splits_path,
            shift=shift,
            pred_model=exp_dataloader.exp_version.pred_model,
            random_type=random_type,
            seed=exp_dataloader.exp_version.version_params["seed"],
        )
