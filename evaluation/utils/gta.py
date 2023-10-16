import cv2
import numpy as np
import uncertainty_modeling.data.cityscapes_labels as cs_labels


def pred_seg_loading(pred_seg_path):
    mask_color = cv2.imread(str(pred_seg_path), -1)
    mask_color = cv2.cvtColor(mask_color, cv2.COLOR_BGR2RGB)
    pred_seg = np.apply_along_axis(
        lambda x: cs_labels.color2trainId.get(tuple(x), 128), axis=-1, arr=mask_color
    )
    return pred_seg


def gt_unc_map(image_id, dataloader):
    idx = dataloader.dataset.image_ids.index(image_id)
    label_path = dataloader.dataset.masks[idx]
    label = np.load(str(label_path))
    unc_map = np.zeros_like(label, dtype=np.single)
    label_switches = {
        "sidewalk": 1.0 / 3.0,
        "person": 1.0 / 3.0,
        "car": 1.0 / 3.0,
        "vegetation": 1.0 / 3.0,
        "road": 1.0 / 3.0,
    }
    for c, p in label_switches.items():
        init_id = cs_labels.name2trainId[c]

        # mean = (1-p) * 0 + p * 1 = p
        mean = p
        variance = (1 - p) * np.square(0 - mean) + p * np.square(1 - mean)
        unc_map[label == init_id] = variance
    unc_map = np.swapaxes(unc_map, 0, 1)
    return unc_map
