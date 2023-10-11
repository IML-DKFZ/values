from itertools import product
from pathlib import Path

import hydra
from omegaconf import ListConfig

from experiment_version import ExperimentVersion
from experiment_dataloader import ExperimentDataloader
from pydantic.utils import deep_update


class EvalExperiments:
    def __init__(self, config):
        # base path is the path to the first experiment cycle
        self.base_path = Path(config.base_path)
        self.second_cycle_path = (
            config.second_cycle_path if "second_cycle_path" in config.keys() else None
        )
        self.versions = self._init_versions(config)
        self.tasks = config.tasks
        self.config = config
        return

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
                version_params["second_cycle_path"] = self.second_cycle_path
                version_params.update(
                    dict(experiment.prediction_models[version_params["pred_model"]])
                )
                exp_version = ExperimentVersion(**version_params)
                versions.append(exp_version)
        return versions

    def analyse_accumulated(self, task_params):
        # This is only used if the results are accumulated across multiple versions
        results_dict_task = {}
        for version in self.versions:
            if "datasets" in task_params.keys():
                dataset_splits = task_params["datasets"]
            else:
                dataset_splits = [None]
            for dataset_split in dataset_splits:
                exp_dataloader = ExperimentDataloader(version, dataset_split)
                results_dict = hydra.utils.instantiate(
                    task_params.function,
                    exp_dataloader=exp_dataloader,
                    _recursive_=False,
                )
                results_dict_task = deep_update(results_dict_task, results_dict)
        hydra.utils.instantiate(
            task_params.postprocess_function,
            results_dict=results_dict_task,
            _recursive_=False,
        )

    def analyse_single_version(self, task_params):
        for version in self.versions:
            if "datasets" in task_params.keys():
                dataset_splits = task_params["datasets"]
            else:
                dataset_splits = [None]
            for dataset_split in dataset_splits:
                exp_dataloader = ExperimentDataloader(version, dataset_split)
                hydra.utils.instantiate(
                    task_params.function,
                    exp_dataloader=exp_dataloader,
                    _recursive_=False,
                )

    def analyse_subtasks(self, tasks):
        for subtask_params in tasks:
            accumulated = (
                subtask_params.accumulated
                if "accumulated" in subtask_params.keys()
                else False
            )
            if accumulated:
                self.analyse_accumulated(task_params=subtask_params)
            else:
                self.analyse_single_version(task_params=subtask_params)

    def analyse(self):
        for task in self.tasks:
            task_params = self.config.task_params[task]
            if type(self.config.task_params[task]) == ListConfig:
                self.analyse_subtasks(task_params)
            else:
                accumulated = (
                    task_params.accumulated
                    if "accumulated" in task_params.keys()
                    else False
                )
                if accumulated:
                    self.analyse_accumulated(task_params=task_params)
                else:
                    self.analyse_single_version(task_params=task_params)
                print(task)
        return


@hydra.main(config_path="configs", config_name="eval_config_lidc", version_base=None)
def main(eval_config):
    evaluator = EvalExperiments(eval_config)
    evaluator.analyse()


if __name__ == "__main__":
    main()
