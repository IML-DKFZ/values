import os.path
from itertools import product
from pathlib import Path

import hydra
from experiment_version import ExperimentVersion


class EvalExperiments:
    def __init__(self, config):
        # base path is the path to the first experiment cycle
        self.base_path = Path(config.base_path)
        self.second_cycle_path = (
            config.second_cycle_path if "second_cycle_path" in config.keys() else None
        )
        self.versions = self._init_versions(config)
        return

    def _init_versions(self, config):
        versions = []
        filtered_config = [
            [(key, v) for v in values]
            for key, values in config.experiments.iter_params.items()
        ]
        for params in product(*filtered_config):
            version_params = {i[0]: i[1] for i in params}
            exp_config = dict(config.experiments)
            exp_config.pop("iter_params")
            version_params.update(exp_config)
            version_params["base_path"] = self.base_path
            version_params["second_cycle_path"] = self.second_cycle_path
            version_params.update(
                dict(config.prediction_models[version_params["pred_model"]])
            )
            exp_version = ExperimentVersion(**version_params)
            versions.append(exp_version)
        return versions

    def analyse(self):
        for version in self.versions:
            print(version.exp_path)
            print(os.path.isdir(version.exp_path))
        return


@hydra.main(
    config_path="configs", config_name="eval_config_toy_seed123", version_base=None
)
def main(eval_config):
    evaluator = EvalExperiments(eval_config)
    evaluator.analyse()


if __name__ == "__main__":
    main()
