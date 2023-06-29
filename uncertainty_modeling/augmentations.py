from typing import Dict, Any

import albumentations as A
import numpy as np

import uncertainty_modeling.data.cityscapes_labels as cs_labels


class StochasticLabelSwitches(A.BasicTransform):
    def __init__(self, always_apply=False, p=0.5, n_reference_samples: int = 1):
        super(StochasticLabelSwitches, self).__init__(always_apply, p)
        self._name2id = cs_labels.name2trainId
        self._label_switches = {
            "sidewalk": 1.0 / 3.0,
            "person": 1.0 / 3.0,
            "car": 1.0 / 3.0,
            "vegetation": 1.0 / 3.0,
            "road": 1.0 / 3.0,
        }
        self.n_reference_samples = n_reference_samples

    def apply(self, img, **params):
        return img

    def apply_to_mask(self, mask, **params):
        masks = []
        for reference in range(self.n_reference_samples):
            mask_copy = mask.copy()
            for c, p in self._label_switches.items():
                init_id = self._name2id[c]
                final_id = self._name2id[c + "_2"]
                switch_instances = np.random.binomial(1, p, 1)

                if switch_instances[0]:
                    mask_copy[mask_copy == init_id] = final_id
            masks.append(mask_copy)
        if len(masks) > 1:
            return np.array(masks)
        else:
            return masks[0]

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    def get_transform_init_args_names(self):
        return ()

    @property
    def targets(self):
        return {"mask": self.apply_to_mask}
