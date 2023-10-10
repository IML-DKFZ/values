import json
import os
from pathlib import Path
import numpy as np
from evaluation.experiment_dataloader import ExperimentDataloader
from itertools import chain


def calculate_foreground_quantile_image(image):
    foreground = np.count_nonzero(image)
    return 1 - (foreground / image.size)


def get_foreground_quantile(exp_dataloader: ExperimentDataloader):
    print(exp_dataloader.dataset_path)
    print(os.path.isdir(exp_dataloader.dataset_path))
    quantile_dict = {exp_dataloader.exp_version.pred_model: {}}
    all_quantiles = []
    for image_id in exp_dataloader.image_ids:
        pred_segs = exp_dataloader.get_pred_segs(image_id)
        for pred_seg in pred_segs:
            perc = calculate_foreground_quantile_image(pred_seg)
            all_quantiles.append(perc)
    quantile_dict[exp_dataloader.exp_version.pred_model][
        exp_dataloader.exp_version.version_name
    ] = all_quantiles
    return quantile_dict


def save_foreground_quantiles(results_dict, save_path):
    methods_results_dict = {}
    for method, versions in results_dict.items():
        method_mean = np.mean(list(chain.from_iterable(versions.values())))
        methods_results_dict[method] = method_mean
    if not os.path.isfile(save_path):
        save_path = Path(save_path) / "quantile_analysis.json"
    print(save_path)
    # save_path_ref = "/home/kckahl/Refactoring/Uncertainty/ToyExperimentOOD/AleatoricTrain/quantile_analysis.json"
    save_path_ref = "/home/kckahl/Refactoring/LIDC-IDRI/ActiveLearning/FirstCycle/quantile_analysis.json"
    with open(save_path_ref, "w") as f:
        json.dump(methods_results_dict, f, indent=2)
