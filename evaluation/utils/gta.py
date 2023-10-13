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
