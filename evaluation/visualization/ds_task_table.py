import json
from itertools import product, groupby
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

import hydra

from evaluation.experiment_version import ExperimentVersion


class DsTaskTable:
    def __init__(self, config):
        # base path is the path to the first experiment cycle
        self.base_path = Path(config.base_path)
        self.versions = self._init_versions(config)
        self.grouped_versions = self._group_versions("seed")
        self.split_param = config.split_param if "split_param" in config else None
        self.ds_tasks = config.ds_tasks

    def _init_versions(self, config):
        versions = []
        for experiment in config.experiments:
            filtered_config = [
                [(key, v) for v in values]
                for key, values in experiment.iter_params.items()
            ]
            for params in product(*filtered_config):
                version_params = {i[0]: i[1] for i in params}
                exp_config = dict(experiment)
                exp_config.pop("iter_params")
                version_params.update(exp_config)
                version_params["base_path"] = self.base_path
                version_params.update(
                    dict(experiment.prediction_models[version_params["pred_model"]])
                )
                exp_version = ExperimentVersion(**version_params)
                versions.append(exp_version)
        return versions

    def _key_func(self, version, version_param):
        naming_scheme_group = version.naming_scheme_version.replace(
            f"{version_param}{{{version_param}}}", ""
        )
        naming_scheme_resolved = naming_scheme_group.format(**version.version_params)
        return version.pred_model, naming_scheme_resolved

    def _group_versions(self, version_param):
        # version should be grouped if all version params except for version_param are the same
        grouped_objects = []
        for _, group in groupby(
            self.versions, key=lambda x: self._key_func(x, version_param)
        ):
            group_list = list(group)
            grouped_objects.append(group_list)
        return grouped_objects

    def get_base_df(self, grouped_versions):
        pred_models = []
        unc_types = []
        aggregations = []
        for group in grouped_versions:
            model_repeat = len(group[0].unc_types) * len(group[0].aggregations)
            pred_models.extend([group[0].pred_model] * model_repeat)
            for unc_type in group[0].unc_types:
                unc_types.extend([unc_type] * len(group[0].aggregations))
            aggregations.extend(list(group[0].aggregations) * len(group[0].unc_types))
        base_df_dict = {
            ("", "pred_model"): pred_models,
            ("", "unc_type"): unc_types,
            ("", "aggregation"): aggregations,
        }

        for ds_task, metrics in self.ds_tasks.items():
            for metric_name, metric_probs in metrics.items():
                if metric_probs.dataset_splits is not None:
                    for split in metric_probs.dataset_splits:
                        index = pd.MultiIndex.from_tuples(
                            [(ds_task, f"{metric_name} {split}")]
                        )
                        base_df_dict[index[0]] = None
                else:
                    index = pd.MultiIndex.from_tuples([(ds_task, metric_name)])
                    base_df_dict[index[0]] = None
                    base_df_dict[ds_task, metric_name] = None
        base_df = pd.DataFrame(base_df_dict)
        base_df.set_index(
            [("", "pred_model"), ("", "unc_type"), ("", "aggregation")],
            inplace=True,
        )
        return base_df

    def fill_metric_pred_model(
        self,
        metric_dicts,
        pred_model,
        mean_df,
        std_df,
        ds_task,
        metric_name,
        metric_key,
        dataset_split,
    ):
        metrics = []
        for metric_dict in metric_dicts:
            if "metrics" in metric_dict["mean"]:
                metrics.append(metric_dict["mean"]["metrics"][metric_key])
            else:
                metrics.append(metric_dict["mean"][metric_key])
        metrics = np.array(metrics)
        metric_mean = np.mean(metrics)
        metric_std = np.std(metrics, ddof=1)
        idx = pd.IndexSlice
        metric_full_name = (
            f"{metric_name} {dataset_split}"
            if dataset_split is not None
            else metric_name
        )
        mean_df.loc[idx[pred_model], [(ds_task, metric_full_name)]] = metric_mean
        std_df.loc[idx[pred_model], [(ds_task, metric_full_name)]] = metric_std

    def fill_metric_pred_model_unc_type(
        self,
        metric_dicts,
        pred_model,
        unc_types,
        mean_df,
        std_df,
        ds_task,
        metric_name,
        metric_key,
        dataset_split,
    ):
        for unc_type in unc_types:
            metrics = []
            for metric_dict in metric_dicts:
                if "metrics" in metric_dict["mean"][unc_type]:
                    metrics.append(metric_dict["mean"][unc_type]["metrics"][metric_key])
                else:
                    metrics.append(metric_dict["mean"][unc_type][metric_key])
            metrics = np.array(metrics)
            metric_mean = np.mean(metrics)
            metric_std = np.std(metrics, ddof=1)
            idx = pd.IndexSlice
            metric_full_name = (
                f"{metric_name} {dataset_split}"
                if dataset_split is not None
                else metric_name
            )
            mean_df.loc[
                idx[pred_model, unc_type], [(ds_task, metric_full_name)]
            ] = metric_mean
            std_df.loc[
                idx[pred_model, unc_type], [(ds_task, metric_full_name)]
            ] = metric_std

    def fill_metric_pred_model_unc_type_agg(
        self,
        metric_dicts,
        pred_model,
        unc_types,
        aggregations,
        mean_df,
        std_df,
        ds_task,
        metric_name,
        metric_key,
        dataset_split,
    ):
        for unc_type in unc_types:
            for aggregation in aggregations:
                metrics = []
                for metric_dict in metric_dicts:
                    if "metrics" in metric_dict["mean"][unc_type][aggregation]:
                        metrics.append(
                            metric_dict["mean"][unc_type][aggregation]["metrics"][
                                metric_key
                            ]
                        )
                    else:
                        metrics.append(
                            metric_dict["mean"][unc_type][aggregation][metric_key]
                        )
                metrics = np.array(metrics)
                metric_mean = np.mean(metrics)
                metric_std = np.std(metrics, ddof=1)
                idx = pd.IndexSlice
                metric_full_name = (
                    f"{metric_name} {dataset_split}"
                    if dataset_split is not None
                    else metric_name
                )
                mean_df.loc[
                    idx[pred_model, unc_type, aggregation],
                    [(ds_task, metric_full_name)],
                ] = metric_mean
                std_df.loc[
                    idx[pred_model, unc_type, aggregation],
                    [(ds_task, metric_full_name)],
                ] = metric_std

    def fill_single_metric(
        self,
        mean_df,
        std_df,
        ds_task,
        metric_name,
        metric_probs,
        versions: List[ExperimentVersion],
        dataset_split,
    ):
        metric_dicts = []
        for version in versions:
            if dataset_split is not None:
                metrics_json = (
                    version.exp_path / dataset_split / metric_probs.metrics_file_name
                )
            else:
                metrics_json = version.exp_path / metric_probs.metrics_file_name
            with open(metrics_json) as f:
                metrics = json.load(f)
            metric_dicts.append(metrics)
        pred_model = versions[0].pred_model
        if len(metric_probs.levels) == 1:
            self.fill_metric_pred_model(
                metric_dicts=metric_dicts,
                pred_model=pred_model,
                mean_df=mean_df,
                std_df=std_df,
                ds_task=ds_task,
                metric_name=metric_name,
                metric_key=metric_probs.metrics_key,
                dataset_split=dataset_split,
            )
        elif len(metric_probs.levels) == 2:
            unc_types = versions[0].unc_types
            self.fill_metric_pred_model_unc_type(
                metric_dicts=metric_dicts,
                pred_model=pred_model,
                unc_types=unc_types,
                mean_df=mean_df,
                std_df=std_df,
                ds_task=ds_task,
                metric_name=metric_name,
                metric_key=metric_probs.metrics_key,
                dataset_split=dataset_split,
            )
        elif len(metric_probs.levels) == 3:
            unc_types = versions[0].unc_types
            aggregations = versions[0].aggregations
            if metric_name == "al_improvement":
                unc_types = [
                    unc_type
                    for unc_type in unc_types
                    if unc_type != "aleatoric_uncertainty"
                ]
            self.fill_metric_pred_model_unc_type_agg(
                metric_dicts=metric_dicts,
                pred_model=pred_model,
                unc_types=unc_types,
                aggregations=aggregations,
                mean_df=mean_df,
                std_df=std_df,
                ds_task=ds_task,
                metric_name=metric_name,
                metric_key=metric_probs.metrics_key,
                dataset_split=dataset_split,
            )

    def fill_all_metrics(self, mean_df, std_df, versions: List[ExperimentVersion]):
        for ds_task, metrics in self.ds_tasks.items():
            for metric_name, metric_probs in metrics.items():
                if metric_probs.dataset_splits is not None:
                    for dataset_split in metric_probs.dataset_splits:
                        self.fill_single_metric(
                            mean_df=mean_df,
                            std_df=std_df,
                            ds_task=ds_task,
                            metric_name=metric_name,
                            metric_probs=metric_probs,
                            versions=versions,
                            dataset_split=dataset_split,
                        )
                else:
                    self.fill_single_metric(
                        mean_df=mean_df,
                        std_df=std_df,
                        ds_task=ds_task,
                        metric_name=metric_name,
                        metric_probs=metric_probs,
                        versions=versions,
                        dataset_split=None,
                    )

    def get_unc_measure(self, df_row):
        if df_row.name[0] == "Softmax":
            return "MSR"
        elif df_row.name[0] == "SSN":
            if df_row.name[1] == "predictive_uncertainty":
                return "PE"
            elif df_row.name[1] == "aleatoric_uncertainty":
                return "MI"
            else:
                return "EE"
        else:
            if df_row.name[1] == "predictive_uncertainty":
                return "PE"
            elif df_row.name[1] == "aleatoric_uncertainty":
                return "EE"
            else:
                return "MI"

    def create_single_table(self, grouped_versions):
        mean_df = self.get_base_df(grouped_versions)
        std_df = self.get_base_df(grouped_versions)
        for group in grouped_versions:
            self.fill_all_metrics(mean_df, std_df, group)

        mean_df[("", "unc_measure")] = mean_df.apply(self.get_unc_measure, axis=1)
        mean_df = mean_df.set_index(("", "unc_measure"), append=True)
        mean_df = mean_df.reorder_levels(
            [
                ("", "pred_model"),
                ("", "unc_measure"),
                ("", "unc_type"),
                ("", "aggregation"),
            ]
        )

        std_df[("", "unc_measure")] = std_df.apply(self.get_unc_measure, axis=1)
        std_df = std_df.set_index(("", "unc_measure"), append=True)
        std_df = std_df.reorder_levels(
            [
                ("", "pred_model"),
                ("", "unc_measure"),
                ("", "unc_type"),
                ("", "aggregation"),
            ]
        )
        # multiply by 100 to see more decimals in table
        mean_df = mean_df * 100
        std_df = std_df * 100
        return mean_df, std_df

    def create(self):
        if self.split_param is not None:
            mean_dfs = []
            std_dfs = []
            for split_value in self.split_param.split_values:
                filtered_grouped_versions = []
                for group in self.grouped_versions:
                    if group[0].version_params[self.split_param.name] == split_value:
                        filtered_grouped_versions.append(group)
                mean_df, std_df = self.create_single_table(filtered_grouped_versions)
                mean_dfs.append(mean_df)
                std_dfs.append(std_df)
            mean_df = pd.concat(mean_dfs, keys=self.split_param.split_values)
            mean_df.index.names = [self.split_param.name, *mean_df.index.names[1:]]
            std_df = pd.concat(std_dfs, keys=self.split_param.split_values)
            std_df.index.names = [self.split_param.name, *std_df.index.names[1:]]
        else:
            mean_df, std_df = self.create_single_table(self.grouped_versions)
        if "Dropout-Final" in mean_df.index.levels[0]:
            mean_df.rename({"Dropout-Final": "Dropout"}, axis=0, level=0, inplace=True)
            std_df.rename({"Dropout-Final": "Dropout"}, axis=0, level=0, inplace=True)
        return mean_df, std_df

    def format_mean_std(self, mean, std):
        mean = mean.astype(float).round(2).astype(str)
        std = std.astype(float).round(2).astype(str)
        return mean.combine(std, lambda x, y: f"{x}±{y}")

    def apply_background_gradient(
        self, styler, cell, reverse=False, results_df=None, split_feature=None
    ):
        if split_feature is None:
            if reverse:
                # reverse means higher scores are better
                # gmap = (results_df[cell]).mul(-1).tolist()
                gmap = (results_df[cell]).mul(-1).tolist()
            else:
                gmap = (results_df[cell]).tolist()
            styler.background_gradient(
                axis=0,
                cmap="YlOrRd",
                gmap=gmap,
                subset=pd.IndexSlice[pd.IndexSlice[:, :, :], [cell]],
            )
        else:
            if reverse:
                # reverse means higher scores are better
                gmap = (results_df.loc[split_feature, cell]).mul(-1).tolist()
            else:
                gmap = (results_df.loc[split_feature, cell]).tolist()
            styler.background_gradient(
                axis=0,
                cmap="YlOrRd",
                gmap=gmap,
                subset=pd.IndexSlice[pd.IndexSlice[split_feature], [cell]],
            )

    def format_color(self, styler, gradient_cells, gradient_cells_reverse, mean_df):
        if self.split_param is not None:
            for split_value in self.split_param.split_values:
                for cell in gradient_cells_reverse:
                    # if cell not in cells_drop:
                    if cell in mean_df:
                        self.apply_background_gradient(
                            styler,
                            cell,
                            reverse=True,
                            results_df=mean_df,
                            split_feature=split_value,
                        )
                for cell in gradient_cells:
                    # if cell not in cells_drop:
                    if cell in mean_df:
                        self.apply_background_gradient(
                            styler,
                            cell,
                            reverse=False,
                            results_df=mean_df,
                            split_feature=split_value,
                        )
        else:
            for cell in gradient_cells_reverse:
                # if cell not in cells_drop:
                if cell in mean_df:
                    self.apply_background_gradient(
                        styler, cell, reverse=True, results_df=mean_df
                    )
            for cell in gradient_cells:
                # if cell not in cells_drop:
                if cell in mean_df:
                    self.apply_background_gradient(
                        styler, cell, reverse=False, results_df=mean_df
                    )

    def to_latex(self, mean_df, std_df):
        results_df = mean_df.combine(std_df, self.format_mean_std)
        results_df.index.names = [
            name if type(name) == str else name[1] for name in results_df.index.names
        ]
        styler = results_df.style

        gradient_cells = []
        gradient_cells_reverse = []
        column_format = "l|" * (len(results_df.index.names)) + "|"
        for ds_task, task_params in self.ds_tasks.items():
            num_metrics_cols = 0
            for metric, metric_params in task_params.items():
                if metric_params["dataset_splits"] is not None:
                    num_metrics_cols += len(metric_params["dataset_splits"])
                    for split in metric_params["dataset_splits"]:
                        if metric_params["higher_better"]:
                            gradient_cells_reverse.append(
                                (ds_task, f"{metric} {split}")
                            )
                        else:
                            gradient_cells.append((ds_task, f"{metric} {split}"))
                else:
                    num_metrics_cols += 1
                    if metric_params["higher_better"]:
                        gradient_cells_reverse.append((ds_task, metric))
                    else:
                        gradient_cells.append((ds_task, metric))
            column_format += "l|" * num_metrics_cols + "|"
            print()
        column_format = column_format[:-2]

        self.format_color(
            styler=styler,
            gradient_cells=gradient_cells,
            gradient_cells_reverse=gradient_cells_reverse,
            mean_df=mean_df,
        )

        latex = styler.to_latex(
            column_format=column_format,
            multicol_align="c",
            convert_css=True,
            position_float="centering",
            hrules=True,
            clines="skip-last;data",
        )

        latex = latex.replace("_", "\_")
        latex = latex.replace("\\centering", "\\centering \\tiny")
        latex = latex.replace(
            "{\cellcolor[HTML]{000000}} \color[HTML]{F1F1F1} nan±nan",
            "{\cellcolor[HTML]{D3D3D3}}",
        )

        # formatting of hlines (make thicker)
        if self.split_param is None:
            num_cols = len(results_df.columns) + len(results_df.index.names)
            latex = latex.replace(
                f"\\cline{{1-{num_cols}}} \\cline{{2-{num_cols}}} \\cline{{3-{num_cols}}}\n\\bottomrule",
                f"\\bottomrule",
            )
            latex = latex.replace(
                f"\\cline{{1-{num_cols}}} \\cline{{2-{num_cols}}} \\cline{{3-{num_cols}}}",
                f"\\cmidrule[2pt]{{1-{num_cols}}}",
            )
        else:
            num_cols = len(results_df.columns) + len(results_df.index.names)
            latex = latex.replace(
                f"\\cline{{1-{num_cols}}} \\cline{{2-{num_cols}}} \\cline{{3-{num_cols}}} \\cline{{4-{num_cols}}}\n\\bottomrule",
                f"\\bottomrule",
            )
            latex = latex.replace(
                f"\\cline{{1-{num_cols}}} \\cline{{2-{num_cols}}} \\cline{{3-{num_cols}}} \\cline{{4-{num_cols}}}",
                f"\\cmidrule[1.5pt]{{1-{num_cols}}} \\morecmidrules \\cmidrule[1.5pt]{{1-{num_cols}}}",
            )
            latex = latex.replace(
                f"\\cline{{2-{num_cols}}} \\cline{{3-{num_cols}}} \\cline{{4-{num_cols}}}",
                f"\\cmidrule[2pt]{{2-{num_cols}}}",
            )
        print(latex)
        return


@hydra.main(config_path="../configs", config_name="table_config_gta", version_base=None)
def main(table_config):
    table = DsTaskTable(table_config)
    mean_df, std_df = table.create()
    table.to_latex(mean_df, std_df)


if __name__ == "__main__":
    main()
