import json
import os
from pathlib import Path
import numpy as np
from medpy.io import load

from evaluation.experiment_dataloader import ExperimentDataloader
from itertools import chain


def calculate_foreground_quantile_image(image):
    foreground = np.count_nonzero(image)
    return 1 - (foreground / image.size)


def get_foreground_quantile(exp_dataloader: ExperimentDataloader):
    print(exp_dataloader.dataset_path)
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
    with open(save_path, "w") as f:
        json.dump(methods_results_dict, f, indent=2)


def threshold_images_paths(exp_dataloader: ExperimentDataloader):
    unc_image_path_dict = {
        exp_dataloader.exp_version.pred_model: {
            exp_dataloader.exp_version.version_name: {}
        }
    }
    for unc_type in exp_dataloader.exp_version.unc_types:
        uncertainty_path = exp_dataloader.unc_path_dict[unc_type]
        unc_image_path_dict[exp_dataloader.exp_version.pred_model][
            exp_dataloader.exp_version.version_name
        ][unc_type] = []
        for image_id in exp_dataloader.image_ids:
            unc_image_path_dict[exp_dataloader.exp_version.pred_model][
                exp_dataloader.exp_version.version_name
            ][unc_type].append(
                uncertainty_path / f"{image_id}{exp_dataloader.exp_version.unc_ending}"
            )
    return unc_image_path_dict


def calculate_threshold_image(quantile_path: Path, image: np.array, method: str):
    with open(quantile_path) as f:
        all_quantiles = json.load(f)
    print(all_quantiles[method])
    threshold = np.quantile(image, all_quantiles[method])
    return threshold


def find_threshold(results_dict, quantile_path, save_path):
    if not os.path.isfile(quantile_path):
        quantile_path = Path(quantile_path) / "quantile_analysis.json"
    if not os.path.isfile(save_path):
        save_path = Path(save_path) / "threshold_analysis.json"
    print(quantile_path)
    print(save_path)
    pred_model_path_dict = {}
    for pred_model, versions in results_dict.items():
        pred_model_path_dict[pred_model] = {}
        for version, uncs in versions.items():
            for unc, paths in uncs.items():
                if unc not in pred_model_path_dict[pred_model]:
                    pred_model_path_dict[pred_model][unc] = []
                pred_model_path_dict[pred_model][unc].extend(paths)
    threshold_dict = {}
    for pred_model, uncs in pred_model_path_dict.items():
        threshold_dict[pred_model] = {}
        for unc, paths in uncs.items():
            unc_images = []
            for path in paths:
                unc_image, _ = load(path)
                unc_images.append(unc_image)
            threshold = calculate_threshold_image(np.array(unc_images), pred_model)
            print(f"Mean {unc.split('_')[0]} threshold: {threshold}")
            threshold_dict[pred_model][
                f"Mean {unc.split('_')[0]} threshold"
            ] = threshold
    all_aleatoric = []
    all_epistemic = []
    all_predictive = []
    for key, value in threshold_dict.items():
        if key != "Softmax":
            all_aleatoric.append(value["Mean aleatoric threshold"])
            all_epistemic.append(value["Mean epistemic threshold"])
        all_predictive.append(value["Mean predictive threshold"])
    mean_dict = {
        "Mean aleatoric threshold": np.mean(all_aleatoric),
        "Mean epistemic threshold": np.mean(all_epistemic),
        "Mean predictive threshold": np.mean(all_predictive),
    }
    threshold_dict["Mean"] = mean_dict
    with open(
        save_path,
        "w",
    ) as f:
        json.dump(threshold_dict, f, indent=2)
