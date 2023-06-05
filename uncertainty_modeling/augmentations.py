from typing import Dict, Any

import albumentations as A
import numpy as np

import uncertainty_modeling.data.cityscapes_labels as cs_labels


class StochasticLabelSwitches(A.BasicTransform):
    def __init__(self, always_apply=False, p=0.5):
        super(StochasticLabelSwitches, self).__init__(always_apply, p)
        self._name2id = cs_labels.name2trainId
        self._label_switches = {
            "sidewalk": 1.0 / 3.0,
            "person": 1.0 / 3.0,
            "car": 1.0 / 3.0,
            "vegetation": 1.0 / 3.0,
            "road": 1.0 / 3.0,
        }

    def apply(self, img, **params):
        return img

    def apply_to_mask(self, mask, **params):
        for c, p in self._label_switches.items():
            init_id = self._name2id[c]
            final_id = self._name2id[c + "_2"]
            switch_instances = np.random.binomial(1, p, 1)

            if switch_instances[0]:
                mask[mask == init_id] = final_id
        return mask

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    def get_transform_init_args_names(self):
        return ()

    @property
    def targets(self):
        return {"mask": self.apply_to_mask}
