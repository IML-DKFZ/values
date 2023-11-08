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
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform

from uncertainty_modeling.data_carrier_3D import DataCarrier3D
from uncertainty_modeling.models.ssn_unet3D_module import SsnUNet3D
from loss_modules import SoftDiceLoss
from main import set_seed
from tqdm import tqdm


def test_cli(config_file: str = None) -> Namespace:
    """
    Set the arguments for testing
    Args:
        config_file: optional, path to default arguments for testing.

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
        "--exp_name",
        type=str,
        default=None,
        help="If given, uses this string as experiment name. "
        "Otherwise, the experiment name will be inferred from the checkpoint.",
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
        "--n_reference_samples",
        type=int,
        default=5,
        help="Number of generated reference samples if samples are simulated",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=12,
        nargs="?",
        help="Size of the test batches to pass. If specified without number, uses same batch size as used in training.",
    )
    parser.add_argument(
        "--test_split",
        type=str,
        default="id",
        help="The key of the test split to use for prediction. If 'unlabeled', uses both, the id and ood unlabeled data",
    )
    parser.add_argument(
        "--test_time_augmentations", "-tta", dest="tta", action="store_true"
    )

    if config_file is not None:
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
    print(args.test_split)
    subject_ids = splits[fold][args.test_split]

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
    elif test_split == "train":
        subject_ids = splits[fold]["train"]
    else:
        subject_ids = splits[fold]["{}_test".format(test_split)]

    test_data_dir = os.path.join(data_input_dir, "preprocessed")
    return test_data_dir, subject_ids


def load_models_from_checkpoint(
    checkpoints: List[Dict], device="cpu"
) -> List[nn.Module]:
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
        all_models.append(model.to(device))
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


def calculate_ged(
    output_softmax: torch.Tensor,
    ground_truth: torch.Tensor,
    ignore_index: int = 0,
    ged_only=False,
) -> Dict:
    gt_repeat = torch.repeat_interleave(ground_truth, output_softmax.shape[0], 0)
    pred_repeat = output_softmax.repeat(
        ground_truth.shape[0], *((output_softmax.ndim - 1) * [1])
    )
    dist_gt_pred_2 = 1 - dice(
        pred_repeat,
        gt_repeat,
        ignore_index=ignore_index,
    )
    pred_1_repeat = torch.repeat_interleave(output_softmax, output_softmax.shape[0], 0)
    pred_1_repeat = torch.argmax(pred_1_repeat, dim=1)
    pred_2_repeat = output_softmax.repeat(
        output_softmax.shape[0], *((output_softmax.ndim - 1) * [1])
    )
    pred_2_repeat = torch.argmax(pred_2_repeat, dim=1)
    dist_pred_pred_2 = 1 - dice(
        pred_1_repeat,
        pred_2_repeat,
        ignore_index=ignore_index if ignore_index == 0 else None,
    )
    gt_1_repeat = torch.repeat_interleave(ground_truth, ground_truth.shape[0], 0)
    gt_2_repeat = ground_truth.repeat(
        ground_truth.shape[0], *((ground_truth.ndim - 1) * [1])
    )
    if torch.any(gt_1_repeat == ignore_index):
        dist_gt_gt_2 = 1 - dice(gt_1_repeat, gt_2_repeat, ignore_index=ignore_index)
    else:
        dist_gt_gt_2 = 1 - dice(gt_1_repeat, gt_2_repeat)
    ged = 2 * dist_gt_pred_2 - dist_pred_pred_2 - dist_gt_gt_2

    if ground_truth.shape[0] > 1 and not ged_only:
        max_dice_rater = []
        for seg_idx in range(ground_truth.shape[0]):
            gt_seg = torch.unsqueeze(ground_truth[seg_idx], 0).type(torch.LongTensor)
            max_dice = torch.tensor(0, dtype=torch.float)
            for pred_idx in range(output_softmax.shape[0]):
                pred_softmax = torch.unsqueeze(output_softmax[pred_idx], 0).type(
                    torch.FloatTensor
                )
                dice_score = dice(pred_softmax, gt_seg, ignore_index=ignore_index)
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
                dice_score = dice(pred_softmax, gt_seg, ignore_index=ignore_index)
                if dice_score > max_dice:
                    max_dice = dice_score
            dice_sum += max_dice
        min_over_preds = dice_sum / output_softmax.shape[0]

    # ged_v2 = ged + dist_mean
    ged_dict = {}
    ged_dict["ged"] = ged.item()
    if ground_truth.shape[0] > 1 and not ged_only:
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
    tta: bool = False,
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
            model = model.double().to("cuda")
            if tta:
                input["data"] = input["data"].copy()
                noise_to_tensor = Compose([GaussianNoiseTransform(), NumpyToTensor()])
                input_noise_tensor = noise_to_tensor(**input)
                flip_dims = [(2,), (3,), (4,), (2, 3), (2, 4), (3, 4), (2, 3, 4)]
                sigma = None
                n_pred = 2 * len(flip_dims) + 2
                for x in [input_tensor["data"], input_noise_tensor["data"]]:
                    output = model.forward(x.double().to("cuda"))
                    output_softmax = F.softmax(output, dim=1).to("cpu")
                    test_datacarrier.concat_data(
                        batch=input_tensor,
                        softmax_pred=output_softmax,
                        n_pred=n_pred * len(models),
                        pred_idx=pred_idx,
                        sigma=sigma,
                    )
                    pred_idx += 1
                    for flip_dim in flip_dims:
                        output = torch.flip(
                            model.forward(torch.flip(x.to("cuda"), flip_dim)), flip_dim
                        )
                        output_softmax = F.softmax(output, dim=1).to("cpu")
                        test_datacarrier.concat_data(
                            batch=input_tensor,
                            softmax_pred=output_softmax,
                            n_pred=n_pred * len(models),
                            pred_idx=pred_idx,
                            sigma=sigma,
                        )
                        pred_idx += 1
            else:
                if hasattr(model, "aleatoric_loss") and model.aleatoric_loss == True:
                    n_pred = n_aleatoric_samples
                    mu, s = model.forward(input_tensor["data"].double().to("cuda"))
                    sigma = torch.exp(s / 2)
                for pred in range(n_pred):
                    if (
                        hasattr(model, "aleatoric_loss")
                        and model.aleatoric_loss == True
                    ):
                        epsilon = torch.randn(s.size())
                        output = mu + sigma * epsilon
                        output_softmax = F.softmax(output, dim=1)
                    else:
                        output = model.forward(input_tensor["data"].double().to("cuda"))
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


def calculate_uncertainty(softmax_preds: torch.Tensor, ssn: bool = False):
    uncertainty_dict = {}
    # softmax_preds = torch.from_numpy(softmax_preds)
    mean_softmax = torch.mean(softmax_preds, dim=0)
    pred_entropy = torch.zeros(*softmax_preds.shape[2:], device=mean_softmax.device)
    for y in range(mean_softmax.shape[0]):
        pred_entropy_class = mean_softmax[y] * torch.log(mean_softmax[y])
        nan_pos = torch.isnan(pred_entropy_class)
        pred_entropy[~nan_pos] += pred_entropy_class[~nan_pos]
    pred_entropy *= -1
    expected_entropy = torch.zeros(
        softmax_preds.shape[0], *softmax_preds.shape[2:], device=softmax_preds.device
    )
    for pred in range(softmax_preds.shape[0]):
        entropy = torch.zeros(*softmax_preds.shape[2:], device=softmax_preds.device)
        for y in range(softmax_preds.shape[1]):
            entropy_class = softmax_preds[pred, y] * torch.log(softmax_preds[pred, y])
            nan_pos = torch.isnan(entropy_class)
            entropy[~nan_pos] += entropy_class[~nan_pos]
        entropy *= -1
        expected_entropy[pred] = entropy
    expected_entropy = torch.mean(expected_entropy, dim=0)
    mutual_information = pred_entropy - expected_entropy
    uncertainty_dict["pred_entropy"] = pred_entropy
    if not ssn:
        uncertainty_dict["aleatoric_uncertainty"] = expected_entropy
        uncertainty_dict["epistemic_uncertainty"] = mutual_information
    else:
        print("mutual information is aleatoric unc")
        uncertainty_dict["aleatoric_uncertainty"] = mutual_information
        uncertainty_dict["epistemic_uncertainty"] = expected_entropy
    # value["softmax_pred"] = np.mean(value["softmax_pred"], axis=0)
    return uncertainty_dict


def calculate_one_minus_msr(softmax_pred: torch.Tensor):
    uncertainty_dict = {}
    max_softmax = softmax_pred.max(dim=0)[0]
    uncertainty_dict["pred_entropy"] = 1 - max_softmax
    return uncertainty_dict


def caculcate_uncertainty_multiple_pred(
    test_datacarrier: DataCarrier3D, ssn: bool = False
) -> None:
    for key, value in test_datacarrier.data.items():
        softmax_preds = torch.from_numpy(value["softmax_pred"])
        value.update(calculate_uncertainty(softmax_preds, ssn))
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
                ),
                dtype=np.intc,
            )
            softmax_pred = np.asarray(
                value["softmax_pred"]
                / np.stack(
                    [np.clip(value["num_predictions"], 1, None)]
                    * value["softmax_pred"].shape[0]
                )
            )
            ged_dict = calculate_ged(
                torch.from_numpy(softmax_pred).to("cuda"),
                torch.from_numpy(gt).to("cuda"),
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
    exp_name = hparams["exp_name"] if args.exp_name is None else args.exp_name
    if "shift_feature" in hparams["datamodule"]:
        test_datacarrier.save_data(
            root_dir=save_dir,
            exp_name=exp_name,
            version=hparams["version"],
            org_data_path=os.path.join(data_input_dir, "images"),
            test_split=args.test_split,
        )
    else:
        if args.test_data_dir is not None:
            org_data_path = None
        else:
            if args.test_split == "val" or args.test_split == "train":
                imagesDir = "imagesTr"
            else:
                imagesDir = "imagesTs"
            org_data_path = os.path.join(
                data_input_dir, hparams["datamodule"]["dataset_name"], imagesDir
            )
        test_datacarrier.save_data(
            root_dir=save_dir,
            exp_name=exp_name,
            version=hparams["version"],
            org_data_path=org_data_path,
            test_split=args.test_split,
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

    set_seed(hparams["seed"])
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
    if args.test_split == "val" or args.test_split == "train":
        test = False
    else:
        test = True
    data_samples = get_val_test_data_samples(
        base_dir=test_data_dir,
        subject_ids=subject_ids,
        test=test,
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
        print(ssn)
    elif "n_aleatoric_samples" in hparams:
        test_datacarrier = predict_cases(
            test_datacarrier,
            data_samples,
            models,
            args.n_pred,
            hparams["n_aleatoric_samples"],
            tta=args.tta,
        )
    else:
        test_datacarrier = predict_cases(
            test_datacarrier, data_samples, models, args.n_pred, tta=args.tta
        )
    if args.n_pred > 1 or len(models) > 1 or args.tta:
        caculcate_uncertainty_multiple_pred(test_datacarrier, ssn)
    calculate_metrics(test_datacarrier)
    save_results(test_datacarrier, hparams, args)


if __name__ == "__main__":
    arguments = test_cli()
    run_test(arguments)
