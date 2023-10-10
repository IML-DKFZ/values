class ExperimentVersion:
    def __init__(
        self,
        base_path,
        second_cycle_path,
        naming_scheme_version,
        pred_model,
        unc_types,
        aggregations,
        naming_scheme_pred_model="{pred_model}",
        **kwargs
    ):
        self.version_name = self._build_version_name(
            naming_scheme_version=naming_scheme_version, **kwargs
        )
        self.exp_path = (
            base_path
            / naming_scheme_pred_model.format(pred_model=pred_model, **kwargs)
            / "test_results"
            / self.version_name
        )
        return

    def _build_version_name(self, naming_scheme_version: str, **kwargs):
        return naming_scheme_version.format(**kwargs)
