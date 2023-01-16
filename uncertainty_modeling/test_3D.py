import os
import pickle
import random
from typing import Dict, Tuple, List

import hydra
import yaml
from collections import OrderedDict
from argparse import Namespace, ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchmetrics.functional import dice

from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.utility_transforms import NumpyToTensor

from uncertainty_modeling.data_carrier_3D import DataCarrier3D
from uncertainty_modeling.models.ssn_unet3D_module import SsnUNet3D
from loss_modules import SoftDiceLoss
from tqdm import tqdm


def test_cli(config_file: str = "configs/test_vnet_defaults.yml") -> Namespace:
    """
    Set the arguments for testing
    Args:
        config_file: path to default arguments for testing. Defaults to './test_unet_defaults.yml'

    Returns:
        args [Namespace]: all arguments needed for testing
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--checkpoint_paths",
        type=str,
        nargs="+",
        help="The path to the checkpoint that should be used to load the model. "
        "Multiple paths can be given for an ensemble prediction. "
        "In this case, configuration parameters like the patch size should be the same for all models "
        "and will be inferred from the checkpoint of the first model.",
    )
    parser.add_argument(
        "-i",
        "--data_input_dir",
        type=str,
        default=None,
        help="If given, dataset root directory to load from. "
        "Otherwise, the input dir will be inferred from the checkpoint. "
        "Specify this if you train and test on different machines (E.g. training on cluster and local testing).",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="If given, uses this string as root directory to save results in. "
        "Otherwise, the save dir will be inferred from the checkpoint. "
        "Specify this if you train and test on different machines (E.g. training on cluster and local testing).",
    )
    parser.add_argument(
        "--test_data_dir",
        type=str,
        default=None,
        help="If given, uses this string as test input."
        "Otherwise, the test samples will be inferred from the test split used in training",
    )
    parser.add_argument(
        "--subject_ids",
        type=list,
        default=None,
        help="If given, only use these subject of the test folder."
        "Otherwise, all images from the test input directory will be used.",
    )
    parser.add_argument(
        "--n_pred",
        type=int,
        default=1,
        help="Number of predictions to make by the model",
    )

    parser.add_argument(
        "--test_split",
        type=str,
        default="id",
        help="The key of the test split to use for prediction. If 'unlabeled', uses both, the id and ood unlabeled data",
    )
    # parser.add_argument("--id", dest="id", action="store_true")
    # parser.add_argument("--ood", dest="id", action="store_false")
    with open(os.path.join(os.path.dirname(__file__), config_file), "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    parser.set_defaults(**config)
    args = parser.parse_args()
    return args


def dir_and_subjects_from_train(
    hparams: Dict, args: Namespace
) -> Tuple[str, List[str]]:
    """
    Get the test samples from the training configuration loaded through the checkpoint
    Args:
        hparams: The hyperparameters from the checkpoint. Needed to infer the path where the test data is as well
                 as the subject ids that are in the test data
        args: Arguments for testing, possibly specifying a data_input_dir

    Returns:
        test_data_dir [str]: The directory which contains the test images
        subject_ids [List[str]]: The list of the subject ids which should be inferred during testing
    """

    data_input_dir = (
        args.data_input_dir
        if args.data_input_dir is not None
        else hparams["data_input_dir"]
    )
    dataset_name = hparams["datamodule"]["dataset_name"]

    with open(os.path.join(data_input_dir, dataset_name, "splits.pkl"), "rb") as f:
        splits = pickle.load(f)
    fold = hparams["datamodule"]["data_fold_id"]
    subject_ids = splits[fold]["test"]

    test_data_dir = os.path.join(data_input_dir, dataset_name, "preprocessed")
    return test_data_dir, subject_ids


def dir_and_subjects_from_train_lidc(
    hparams: Dict, args: Namespace, test_split: str = "id"
) -> Tuple[str, List[str]]:
    """
    Get the test samples from the training configuration loaded through the checkpoint
    Args:
        hparams: The hyperparameters from the checkpoint. Needed to infer the path where the test data is as well
                 as the subject ids that are in the test data
        args: Arguments for testing, possibly specifying a data_input_dir
        id: whether to predict id cases

    Returns:
        test_data_dir [str]: The directory which contains the test images
        subject_ids [List[str]]: The list of the subject ids which should be inferred during testing
    """

    data_input_dir = (
        args.data_input_dir
        if args.data_input_dir is not None
        else hparams["data_input_dir"]
    )
    # dataset_name = hparams["datamodule"]["dataset_name"]
    shift_feature = hparams["datamodule"]["shift_feature"]

    if "splits_path" in hparams["datamodule"].keys():
        if hparams["datamodule"]["splits_path"] is not None:
            if args.data_input_dir is not None:
                splits_path = hparams["datamodule"]["splits_path"].replace(
                    hparams["data_input_dir"], args.data_input_dir
                )
            else:
                splits_path = hparams["datamodule"]["splits_path"]
        else:
            splits_path = os.path.join(
                data_input_dir,
                "splits_{}.pkl".format(shift_feature)
                if shift_feature is not None
                else "all",
            )
    else:
        splits_path = os.path.join(
            data_input_dir,
            "splits_{}.pkl".format(shift_feature)
            if shift_feature is not None
            else "all",
        )

    with open(
        splits_path,
        "rb",
    ) as f:
        splits = pickle.load(f)
    fold = hparams["datamodule"]["data_fold_id"]
    if test_split == "unlabeled":
        subject_ids = splits[fold]["id_unlabeled_pool"]
        subject_ids = np.concatenate((subject_ids, splits[fold]["ood_unlabeled_pool"]))
    elif test_split == "val":
        subject_ids = splits[fold]["val"]
    else:
        subject_ids = splits[fold]["{}_test".format(test_split)]

    test_data_dir = os.path.join(data_input_dir, "preprocessed")
    return test_data_dir, subject_ids


def load_models_from_checkpoint(checkpoints: List[Dict]) -> List[nn.Module]:
    """
    Load the model for the predictions from a checkpoint
    Args:
        checkpoints: The checkpoints to load the model from

    Returns:
        model: The model for the predictions
    """
    all_models = []
    for checkpoint in checkpoints:
        hparams = checkpoint["hyper_parameters"]
        state_dict = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            state_dict[".".join(k.split(".")[1:])] = v
        if "aleatoric_loss" in hparams and hparams["aleatoric_loss"] is not None:
            model = hydra.utils.instantiate(
                hparams["model"], aleatoric_loss=hparams["aleatoric_loss"]
            )
        else:
            model = hydra.utils.instantiate(hparams["model"])
        model.load_state_dict(state_dict=state_dict)
        all_models.append(model)
    return all_models


def calculate_test_metrics(
    output_softmax: torch.Tensor, ground_truth: torch.Tensor
) -> Dict:
    """
    Calculate the metrics for evaluation
    Args:
        output_softmax [torch.Tensor]: The output of the network after applying softmax.
        ground_truth [torch.Tensor]: The ground truth segmentation.

    Returns:
        metrics_dict [Dict]: A dictionary with the calculated metrics
    """
    dice_loss = SoftDiceLoss()
    nll_loss = torch.nn.NLLLoss()

    all_test_loss = []
    all_test_dice = []
    for rater in range(ground_truth.size(dim=0)):
        gt_seg = ground_truth[rater]
        gt_seg = torch.unsqueeze(gt_seg, 0).type(torch.LongTensor)

        test_loss = dice_loss(output_softmax, gt_seg) + nll_loss(
            torch.log(output_softmax), gt_seg
        )
        test_dice = dice(output_softmax, gt_seg, ignore_index=0)
        all_test_loss.append(test_loss.item())
        all_test_dice.append(test_dice.item())
    metrics_dict = {
        "loss": np.mean(np.array(all_test_loss)),
        "dice": np.mean(np.array(all_test_dice)),
    }
    return metrics_dict
    # dice_loss = SoftDiceLoss()
    # nll_loss = torch.nn.NLLLoss()
    # test_loss = dice_loss(output_softmax, ground_truth) + nll_loss(
    #     torch.log(output_softmax), ground_truth
    # )
    # test_dice = dice(output_softmax, ground_truth, ignore_index=0)
    # metrics_dict = {"loss": test_loss.item(), "dice": test_dice.item()}
    # return metrics_dict


def calculate_ged(output_softmax: torch.Tensor, ground_truth: torch.Tensor) -> Dict:
    dist_gt_pred = []
    for seg_idx in range(ground_truth.shape[0]):
        gt_seg = torch.unsqueeze(ground_truth[seg_idx], 0).type(torch.LongTensor)
        for pred_idx in range(output_softmax.shape[0]):
            pred_softmax = torch.unsqueeze(output_softmax[pred_idx], 0).type(
                torch.FloatTensor
            )
            dist = 1 - dice(pred_softmax, gt_seg, ignore_index=0)
            dist_gt_pred.append(dist)
    dist_pred_pred = []
    for pred_idx_1 in range(output_softmax.shape[0]):
        pred_softmax_1 = torch.unsqueeze(output_softmax[pred_idx_1], 0).type(
            torch.FloatTensor
        )
        pred_1 = torch.argmax(pred_softmax_1, dim=1).type(torch.LongTensor)
        for pred_idx_2 in range(output_softmax.shape[0]):
            pred_softmax_2 = torch.unsqueeze(output_softmax[pred_idx_2], 0).type(
                torch.FloatTensor
            )
            pred_2 = torch.argmax(pred_softmax_2, dim=1).type(torch.LongTensor)
            dist = 1 - dice(pred_1, pred_2, ignore_index=0)
            dist_pred_pred.append(dist)
    dist_gt_gt = []
    for seg_idx_1 in range(ground_truth.shape[0]):
        gt_seg_1 = torch.unsqueeze(ground_truth[seg_idx_1], 0).type(torch.LongTensor)
        for seg_idx_2 in range(ground_truth.shape[0]):
            gt_seg_2 = torch.unsqueeze(ground_truth[seg_idx_2], 0).type(
                torch.LongTensor
            )
            dist = 1 - dice(gt_seg_1, gt_seg_2, ignore_index=0)
            dist_gt_gt.append(dist)
    ged = 2 * np.mean(dist_gt_pred) - np.mean(dist_pred_pred) - np.mean(dist_gt_gt)

    if ground_truth.shape[0] > 1:
        max_dice_rater = []
        for seg_idx in range(ground_truth.shape[0]):
            gt_seg = torch.unsqueeze(ground_truth[seg_idx], 0).type(torch.LongTensor)
            max_dice = torch.tensor(0, dtype=torch.float)
            for pred_idx in range(output_softmax.shape[0]):
                pred_softmax = torch.unsqueeze(output_softmax[pred_idx], 0).type(
                    torch.FloatTensor
                )
                dice_score = dice(pred_softmax, gt_seg, ignore_index=0)
                if dice_score > max_dice:
                    max_dice = dice_score
            max_dice_rater.append(max_dice)

        dice_sum = torch.tensor(0, dtype=torch.float)
        for pred_idx in range(output_softmax.shape[0]):
            pred_softmax = torch.unsqueeze(output_softmax[pred_idx], 0).type(
                torch.FloatTensor
            )
            max_dice = torch.tensor(0, dtype=torch.float)
            for seg_idx in range(ground_truth.shape[0]):
                gt_seg = torch.unsqueeze(ground_truth[seg_idx], 0).type(
                    torch.LongTensor
                )
                dice_score = dice(pred_softmax, gt_seg, ignore_index=0)
                if dice_score > max_dice:
                    max_dice = dice_score
            dice_sum += max_dice
        min_over_preds = dice_sum / output_softmax.shape[0]

    # ged_v2 = ged + dist_mean
    ged_dict = {}
    ged_dict["ged"] = ged
    if ground_truth.shape[0] > 1:
        for idx, rater_dist in enumerate(max_dice_rater):
            ged_dict["max dice rater {}".format(idx)] = rater_dist.item()
        ged_dict["max dice pred"] = min_over_preds.item()

    return ged_dict


def predict_cases_ssn(
    test_datacarrier: DataCarrier3D,
    data_samples: List[Dict],
    model: nn.Module,
    n_pred: int = 1,
) -> DataCarrier3D:
    model = model.double()
    for sample in tqdm(data_samples):
        input = test_datacarrier.load_image(sample)
        input["data"] = np.expand_dims(input["data"], axis=0)
        to_tensor = Compose([NumpyToTensor()])
        input_tensor = to_tensor(**input)
        distribution = model.forward(input_tensor["data"])
        output_samples = distribution.sample([n_pred])
        output_samples = output_samples.view(
            [
                n_pred,
                1,
                model.num_classes,
                *input_tensor["data"].size()[-3:],
            ]
        )

        pred_idx = 0
        for output_sample in output_samples:
            output_softmax = F.softmax(output_sample, dim=1)

            test_datacarrier.concat_data(
                batch=input_tensor,
                softmax_pred=output_softmax,
                n_pred=n_pred,
                pred_idx=pred_idx,
                sigma=None,
            )
            pred_idx += 1
    return test_datacarrier


def predict_cases(
    test_datacarrier: DataCarrier3D,
    data_samples: List[Dict],
    models: List[nn.Module],
    n_pred: int = 1,
    n_aleatoric_samples: int = 10,
) -> DataCarrier3D:
    """
    Predict all test cases.
    Args:
        test_datacarrier: The datacarrier to save the data
        data_samples: The samples to predict
        model: The model used for prediction

    Returns:
        test_datacarrier: The datacarrier with the concatenated data
    """
    for sample in tqdm(data_samples):
        input = test_datacarrier.load_image(sample)
        input["data"] = np.expand_dims(input["data"], axis=0)
        to_tensor = Compose([NumpyToTensor()])
        input_tensor = to_tensor(**input)

        pred_idx = 0
        for model in models:
            model = model.double()
            if hasattr(model, "aleatoric_loss") and model.aleatoric_loss == True:
                n_pred = n_aleatoric_samples
                mu, s = model.forward(input_tensor["data"].double())
                sigma = torch.exp(s / 2)
            for pred in range(n_pred):
                if hasattr(model, "aleatoric_loss") and model.aleatoric_loss == True:
                    epsilon = torch.randn(s.size())
                    output = mu + sigma * epsilon
                    output_softmax = F.softmax(output, dim=1)
                else:
                    output = model.forward(input_tensor["data"].double())
                    output_softmax = F.softmax(output, dim=1)
                    sigma = None

                test_datacarrier.concat_data(
                    batch=input_tensor,
                    softmax_pred=output_softmax,
                    n_pred=n_pred * len(models),
                    pred_idx=pred_idx,
                    sigma=sigma,
                )
                pred_idx += 1
    return test_datacarrier


def caculcate_uncertainty_multiple_pred(
    test_datacarrier: DataCarrier3D, ssn: bool = False
) -> None:
    for key, value in test_datacarrier.data.items():
        softmax_preds = torch.from_numpy(value["softmax_pred"])
        mean_softmax = torch.mean(softmax_preds, dim=0)
        pred_entropy = torch.zeros(
            softmax_preds.shape[2],
            softmax_preds.shape[3],
            softmax_preds.shape[4],
        )
        for y in range(mean_softmax.shape[0]):
            pred_entropy += mean_softmax[y] * torch.log(mean_softmax[y])
        pred_entropy *= -1
        expected_entropy = torch.zeros(
            softmax_preds.shape[0],
            softmax_preds.shape[2],
            softmax_preds.shape[3],
            softmax_preds.shape[4],
        )
        for pred in range(softmax_preds.shape[0]):
            entropy = torch.zeros(
                softmax_preds.shape[2],
                softmax_preds.shape[3],
                softmax_preds.shape[4],
            )
            for y in range(softmax_preds.shape[1]):
                entropy += softmax_preds[pred, y, :, :, :] * torch.log(
                    softmax_preds[pred, y, :, :, :]
                )
            entropy *= -1
            expected_entropy[pred] = entropy
        expected_entropy = torch.mean(expected_entropy, dim=0)
        mutual_information = pred_entropy - expected_entropy
        value["pred_entropy"] = pred_entropy
        if not ssn:
            value["aleatoric_uncertainty"] = expected_entropy
            value["epistemic_uncertainty"] = mutual_information
        else:
            value["aleatoric_uncertainty"] = mutual_information
            value["epistemic_uncertainty"] = expected_entropy
        # value["softmax_pred"] = np.mean(value["softmax_pred"], axis=0)
    return


def calculate_metrics(test_datacarrier: DataCarrier3D) -> None:
    """
    Calculate metrics when all slices for all cases have been predicted
    Args:
        test_datacarrier: The datacarrier with the concatenated data
    """
    print("New metrics calculation")
    for key, value in test_datacarrier.data.items():
        mean_softmax_pred = torch.mean(
            torch.from_numpy(
                value["softmax_pred"] / np.clip(value["num_predictions"], 1, None)[0]
            ),
            dim=0,
        )
        mean_softmax_pred = torch.unsqueeze(mean_softmax_pred, 0)
        gt_seg = torch.from_numpy(np.asarray(value["seg"]))
        metrics_dict = calculate_test_metrics(mean_softmax_pred, gt_seg)
        if value["seg"].shape[0] > 1 or value["softmax_pred"].shape[0] > 1:
            gt = np.asarray(
                value["seg"]
                / np.stack(
                    [np.clip(value["num_predictions"], 1, None)[0]]
                    * value["seg"].shape[0]
                )
            )
            softmax_pred = np.asarray(
                value["softmax_pred"]
                / np.stack(
                    [np.clip(value["num_predictions"], 1, None)]
                    * value["softmax_pred"].shape[0]
                )
            )
            ged_dict = calculate_ged(
                torch.from_numpy(softmax_pred), torch.from_numpy(gt)
            )
            metrics_dict.update(ged_dict)
        test_datacarrier.data[key]["metrics"] = metrics_dict


def save_results(
    test_datacarrier: DataCarrier3D, hparams: Dict, args: Namespace
) -> None:
    """
    Save the results of the datacarrier to disc

    Args:
        test_datacarrier: The datacarrier which contains the data to save
        hparams: Dict with hyperparameters of training. Needed to infer the path where to store test results.
        args: Arguments for testing, possibly specifying a data_input_dir and a save_dir
    """
    save_dir = args.save_dir if args.save_dir is not None else hparams["save_dir"]
    data_input_dir = (
        args.data_input_dir
        if args.data_input_dir is not None
        else hparams["data_input_dir"]
    )
    if "shift_feature" in hparams["datamodule"]:
        test_datacarrier.save_data(
            root_dir=save_dir,
            exp_name=hparams["exp_name"],
            version=hparams["version"],
            org_data_path=os.path.join(data_input_dir, "images"),
            test_split=args.test_split,
        )
    else:
        test_datacarrier.save_data(
            root_dir=save_dir,
            exp_name=hparams["exp_name"],
            version=hparams["version"],
            org_data_path=os.path.join(
                data_input_dir, hparams["datamodule"]["dataset_name"], "imagesTs"
            ),
        )
    test_datacarrier.log_metrics()


def run_test(args: Namespace) -> None:
    """
    Run test and save the results in the end
    Args:
        args: Arguments for testing, including checkpoint_path, test_data_dir and subject_ids.
              test_data_dir and subject_ids might be None.
    """
    test_data_dir = args.test_data_dir
    subject_ids = args.subject_ids

    all_checkpoints = []
    for checkpoint_path in args.checkpoint_paths:
        checkpoint = torch.load(checkpoint_path)
        all_checkpoints.append(checkpoint)
    hparams = all_checkpoints[0]["hyper_parameters"]

    # No test data dir specified, so data should be in same input dir as training data and split should be specified
    if test_data_dir is None:
        if "shift_feature" in hparams["datamodule"]:
            test_data_dir, subject_ids = dir_and_subjects_from_train_lidc(
                hparams, args, args.test_split
            )
        else:
            test_data_dir, subject_ids = dir_and_subjects_from_train(hparams, args)

    test_datacarrier = DataCarrier3D()
    if "shift_feature" in hparams["datamodule"]:
        from uncertainty_modeling.lidc_idri_datamodule_3D import (
            get_val_test_data_samples,
        )
    else:
        from uncertainty_modeling.toy_datamodule_3D import get_val_test_data_samples
    data_samples = get_val_test_data_samples(
        base_dir=test_data_dir,
        subject_ids=subject_ids,
        test=True,
        num_raters=hparams["datamodule"]["num_raters"],
        patch_size=hparams["datamodule"]["patch_size"],
        patch_overlap=hparams["datamodule"]["patch_overlap"],
    )

    models = load_models_from_checkpoint(all_checkpoints)
    # data_samples = [data_samples[0]]
    ssn = False
    if isinstance(models[0], SsnUNet3D) and len(models) == 1:
        test_datacarrier = predict_cases_ssn(
            test_datacarrier, data_samples, models[0], args.n_pred
        )
        ssn = True
    elif "n_aleatoric_samples" in hparams:
        test_datacarrier = predict_cases(
            test_datacarrier,
            data_samples,
            models,
            args.n_pred,
            hparams["n_aleatoric_samples"],
        )
    else:
        test_datacarrier = predict_cases(
            test_datacarrier, data_samples, models, args.n_pred
        )
    if args.n_pred > 1 or len(models) > 1:
        caculcate_uncertainty_multiple_pred(test_datacarrier, ssn)
    calculate_metrics(test_datacarrier)
    save_results(test_datacarrier, hparams, args)


if __name__ == "__main__":
    arguments = test_cli()
    random.seed(14)
    run_test(arguments)
