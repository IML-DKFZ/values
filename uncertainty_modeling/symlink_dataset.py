import json
import os

import numpy as np


def create_symlinks(
    dataset_image_dir,
    dataset_labels_dir,
    results_dir,
    file_ending: str = ".png",
    overwrite: bool = True,
):
    results_image_dir = os.path.join(results_dir, "input")
    results_labels_dir = os.path.join(results_dir, "gt_seg")
    os.makedirs(results_image_dir, exist_ok=True)
    os.makedirs(results_labels_dir, exist_ok=True)
    metrics_file = os.path.join(results_dir, "metrics.json")
    with open(metrics_file, "r") as f:
        metrics = json.load(f)
    for image_id in metrics.keys():
        if image_id != "mean":
            for dataset_path, results_path in [
                (dataset_image_dir, results_image_dir),
                (dataset_labels_dir, results_labels_dir),
            ]:
                source_path = os.path.join(dataset_path, f"{image_id}{file_ending}")
                dest_path = os.path.join(results_path, f"{image_id}{file_ending}")
                try:
                    os.symlink(source_path, dest_path)
                except FileExistsError:
                    if overwrite:
                        print("symlink exists but is overwritten")
                        os.remove(dest_path)
                        os.symlink(source_path, dest_path)
                    else:
                        print("symlink exists and is not overwritten")
                        continue


def main():
    gta_dir = "/nvme/GTA/OriginalData/preprocessed"
    gta_image_dir = os.path.join(gta_dir, "images", "vis")
    gta_labels_dir = os.path.join(gta_dir, "labels", "vis")
    results_dir = (
        "/nvme/GTA/Experiments/FirstCycle/Ensemble/test_results/fold0_seed123/id"
    )
    create_symlinks(gta_image_dir, gta_labels_dir, results_dir, overwrite=False)


if __name__ == "__main__":
    # main()
    arr = np.load(
        "/nvme/GTA/Experiments/FirstCycle/Ensemble/test_results/fold0_seed123/ood/pred_prob/frankfurt_000000_000294_0.npz"
    )
    print()
