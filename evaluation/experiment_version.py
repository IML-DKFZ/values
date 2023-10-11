class ExperimentVersion:
    def __init__(
        self,
        base_path,
        second_cycle_path,
        naming_scheme_version,
        pred_model,
        image_ending,
        unc_ending,
        unc_types,
        n_classes=2,
        naming_scheme_pred_model="{pred_model}",
        **kwargs
    ):
        self.pred_model = pred_model
        self.version_name = self._build_version_name(
            naming_scheme_version=naming_scheme_version, **kwargs
        )
        self.base_path = base_path
        self.exp_path = (
            base_path
            / naming_scheme_pred_model.format(pred_model=pred_model, **kwargs)
            / "test_results"
            / self.version_name
        )
        self.image_ending = image_ending
        self.unc_ending = unc_ending
        self.n_classes = n_classes
        self.unc_types = unc_types

    def _build_version_name(self, naming_scheme_version: str, **kwargs):
        return naming_scheme_version.format(**kwargs)
