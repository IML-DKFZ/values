def sort_uncertainties_image_level(uncertainties: dict):
    image_level_dict = {
        key: value["image_level"]["max_score"] for key, value in uncertainties.items()
    }
    image_level_dict = sorted(image_level_dict, key=image_level_dict.get, reverse=True)
    # print(image_level_dict)
    return image_level_dict


def sort_uncertainties_patch_level(uncertainties: dict):
    patch_level_dict = {
        key: value["patch_level"]["max_score"] for key, value in uncertainties.items()
    }
    patch_level_dict = sorted(patch_level_dict, key=patch_level_dict.get, reverse=True)
    # print(patch_level_dict)
    return patch_level_dict


def sort_uncertainties_threshold_level(uncertainties: dict):
    threshold_level_dict = {
        key: value["threshold"]["max_score"] for key, value in uncertainties.items()
    }
    threshold_level_dict = sorted(
        threshold_level_dict, key=threshold_level_dict.get, reverse=True
    )
    # print(threshold_level_dict)
    return threshold_level_dict


def sort_uncertainties(uncertainties: dict, level: str):
    if level == "image_level":
        return sort_uncertainties_image_level(uncertainties)
    elif level == "patch_level":
        return sort_uncertainties_patch_level(uncertainties)
    elif level == "threshold":
        return sort_uncertainties_threshold_level(uncertainties)
    else:
        raise Exception("Uncertainty level not known!")
