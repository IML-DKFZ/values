import json

from evaluation.experiment_dataloader import ExperimentDataloader


def main(exp_dataloader: ExperimentDataloader):
    al_improv_dict = {"mean": {}}
    metrics_path_first_cycle = exp_dataloader.dataset_path / "metrics.json"
    with open(metrics_path_first_cycle) as f:
        metrics_first_cycle = json.load(f)
    if "metrics" in metrics_first_cycle["mean"].keys():
        metrics_key = True
    else:
        metrics_key = False
    if metrics_key:
        dice_first_cycle = metrics_first_cycle["mean"]["metrics"]["dice"]
    else:
        dice_first_cycle = metrics_first_cycle["mean"]["dice"]

    exp_path_second_cycle_random = (
        exp_dataloader.exp_version.second_cycle_path
        / exp_dataloader.exp_version.pred_model
        / "test_results"
        / "random"
        / "random"
        / exp_dataloader.exp_version.version_name
        / exp_dataloader.dataset_split
    )
    metrics_path_second_cycle_random = exp_path_second_cycle_random / "metrics.json"
    with open(metrics_path_second_cycle_random) as f:
        metrics_second_cycle_random = json.load(f)
    if metrics_key:
        dice_second_cycle_random = metrics_second_cycle_random["mean"]["metrics"][
            "dice"
        ]
    else:
        dice_second_cycle_random = metrics_second_cycle_random["mean"]["dice"]

    for unc_type in exp_dataloader.exp_version.unc_types:
        if unc_type == "aleatoric_uncertainty":
            continue
        al_improv_dict["mean"][unc_type] = {}
        for aggregation in exp_dataloader.exp_version.aggregations:
            al_improv_dict["mean"][unc_type][aggregation] = {}
            exp_path_second_cycle_unc = (
                exp_dataloader.exp_version.second_cycle_path
                / exp_dataloader.exp_version.pred_model
                / "test_results"
                / unc_type
                / aggregation
                / exp_dataloader.exp_version.version_name
                / exp_dataloader.dataset_split
            )
            metrics_path_second_cycle_unc = exp_path_second_cycle_unc / "metrics.json"
            with open(metrics_path_second_cycle_unc) as f:
                metrics_second_cycle_unc = json.load(f)
            if metrics_key:
                dice_second_cycle_unc = metrics_second_cycle_unc["mean"]["metrics"][
                    "dice"
                ]
            else:
                dice_second_cycle_unc = metrics_second_cycle_unc["mean"]["dice"]
            rel_improv_unc = (dice_second_cycle_unc / dice_first_cycle) - 1
            rel_improv_random = (dice_second_cycle_random / dice_first_cycle) - 1
            overall_improv = rel_improv_unc - rel_improv_random
            al_improv_dict["mean"][unc_type][aggregation] = {
                "metrics": {"al_improvement": overall_improv}
            }

    save_path = exp_dataloader.dataset_path / "al_improvement.json"
    with open(save_path, "w") as f:
        json.dump(al_improv_dict, f, indent=2)
