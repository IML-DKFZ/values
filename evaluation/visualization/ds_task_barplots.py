import copy
import os
from pathlib import Path

import hydra
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from evaluation.visualization.ds_task_table import DsTaskTable


def generate_barplot(
    ds_task: str,
    metric: str,
    dimension: str,
    dataset_dfs: dict,
    results_plot_dir: Path,
    lower_better: bool = False,
    percent: bool = False,
    df_naming=None,
    coloring=None,
    ordering=None,
    filter_index=None,
):
    all_mean_dfs = []
    all_std_dfs = []
    for df_name, df in dataset_dfs.items():
        if filter_index:
            for dim in filter_index:
                df = df.loc[df.index.get_level_values(("", dim[0])) != dim[1]]
        mean_ds = df.loc[:, (ds_task, metric)].mean()
        df.loc[:, (ds_task, metric)] -= mean_ds
        if lower_better:
            df.loc[:, (ds_task, metric)] *= -1

        if df_naming and df_name in df_naming:
            df_name_plot = df_naming[df_name]
        else:
            df_name_plot = df_name

        mu = round(mean_ds, 2) if percent else round(mean_ds / 100, 2)
        df_mean = (
            df[(ds_task, metric)]
            .groupby(("", dimension))
            .mean()
            .rename(f"{df_name_plot} \n (µ: {round(mu, 2)})")
            .to_frame()
        )
        df_std = (
            df[(ds_task, metric)]
            .groupby(("", dimension))
            .std()
            .rename(f"{df_name_plot} \n (µ: {round(mu, 2)})")
            .to_frame()
        )
        all_mean_dfs.append(df_mean)
        all_std_dfs.append(df_std)
    dfs_mean_concat = pd.concat(all_mean_dfs, axis=1)
    dfs_std_concat = pd.concat(all_std_dfs, axis=1)
    dfs_mean_concat.dropna(inplace=True, how="all")
    dfs_std_concat.dropna(inplace=True, how="all")
    if ordering and dimension in ordering:
        dfs_mean_concat = dfs_mean_concat.reindex(ordering[dimension])
        dfs_std_concat = dfs_std_concat.reindex(ordering[dimension])
    sns.set_style("whitegrid")
    if coloring and dimension in coloring:
        colors = dict(coloring[dimension])
    else:
        colors = None
    ax = dfs_mean_concat.T.plot.bar(
        yerr=dfs_std_concat.T, capsize=4, figsize=(5, 6), color=colors, fontsize=19
    )
    plt.ylabel(" ".join(metric.split(" ")[0].split("_")), fontsize=19)
    plt.xticks(rotation=0)
    ax.set_yticks(ax.get_yticks().tolist())
    if percent:
        ax.set_yticklabels(
            [round(l, 3) if float(l) != 0.0 else "µ" for l in ax.get_yticks().tolist()]
        )
    else:
        ax.set_yticklabels(
            [
                round(l / 100, 3) if float(l) != 0.0 else "µ"
                for l in ax.get_yticks().tolist()
            ]
        )
    plt.axhline(y=0.0, color="black", linestyle="-")
    results_plot_dir = results_plot_dir / dimension
    os.makedirs(results_plot_dir, exist_ok=True)
    results_plot_path = results_plot_dir / f"{'_'.join(metric.lower().split(' '))}.png"
    ax.get_legend().remove()
    plt.tight_layout()
    plt.savefig(results_plot_path)
    plt.close()


@hydra.main(config_path="../configs", config_name="plot_config", version_base=None)
def main(plot_config):
    warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
    pd.set_option("mode.chained_assignment", None)

    dataset_dfs = {}
    for dataset, table_config in plot_config.datasets.items():
        table = DsTaskTable(table_config)
        mean_df, _ = table.create()
        if "split_param" in table_config:
            for split_value in table_config.split_param.split_values:
                mean_df_splitted = mean_df.loc[split_value]
                dataset_dfs[f"{dataset} {split_value.title()}"] = mean_df_splitted
        else:
            dataset_dfs[dataset] = mean_df

    for ds_task, task_config in plot_config.ds_tasks.items():
        for metric, metric_config in task_config.items():
            for dimension in metric_config.levels:
                filter_ds = None
                if "filter" in metric_config:
                    filter_ds = []
                    if dimension in metric_config.filter:
                        for filter_dim, filter_values in metric_config.filter[
                            dimension
                        ].items():
                            for value in filter_values:
                                filter_ds.append((filter_dim, value))
                df_copy = copy.deepcopy(dataset_dfs)
                lower_better = not metric_config.higher_better
                coloring = plot_config.coloring if "coloring" in plot_config else None
                ordering = plot_config.ordering if "ordering" in plot_config else None
                percent = metric_config.percent if "percent" in metric_config else False
                df_naming = (
                    plot_config.df_naming if "df_naming" in plot_config else None
                )
                if metric_config.dataset_splits is None:
                    generate_barplot(
                        ds_task=ds_task,
                        metric=metric,
                        dimension=dimension,
                        dataset_dfs=df_copy,
                        lower_better=lower_better,
                        percent=percent,
                        filter_index=filter_ds,
                        df_naming=df_naming,
                        coloring=coloring,
                        ordering=ordering,
                        results_plot_dir=Path(plot_config.save_path),
                    )
                else:
                    for ds_split in metric_config.dataset_splits:
                        generate_barplot(
                            ds_task=ds_task,
                            metric=f"{metric} {ds_split}",
                            dimension=dimension,
                            dataset_dfs=df_copy,
                            lower_better=lower_better,
                            percent=percent,
                            filter_index=filter_ds,
                            df_naming=df_naming,
                            coloring=coloring,
                            ordering=ordering,
                            results_plot_dir=Path(plot_config.save_path),
                        )


if __name__ == "__main__":
    main()
