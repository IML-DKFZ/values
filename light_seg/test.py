import os
import pickle
from typing import Dict, Tuple, List

import yaml
from collections import OrderedDict
from argparse import Namespace, ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchmetrics.functional.classification import dice_score

from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.spatial_transforms import ResizeTransform
from batchgenerators.transforms.utility_transforms import NumpyToTensor

from light_seg.unet_module import UNet
from data_carrier import DataCarrier
from loss_modules import SoftDiceLoss
from tqdm import tqdm
from torchvision import transforms


def test_cli(config_file: str = "./test_unet_defaults.yml") -> Namespace:
    """
    Set the arguments for testing
    Args:
        config_file: path to default arguments for testing. Defaults to './test_unet_defaults.yml'

    Returns:
        args [Namespace]: all arguments needed for testing
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="The path to the chckpoint that should be used to load the model.",
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

    Returns:
        test_data_dir [str]: The directory which contains the test images
        subject_ids [List[str]]: The list of the subject ids which should be inferred during testing
    """

    data_input_dir = (
        args.data_input_dir
        if args.data_input_dir is not None
        else hparams["data_input_dir"]
    )
    dataset_name = hparams["dataset_name"]

    with open(os.path.join(data_input_dir, dataset_name, "splits.pkl"), "rb") as f:
        splits = pickle.load(f)
    fold = hparams["data_fold_id"]
    subject_ids = splits[fold]["test"]

    test_data_dir = os.path.join(data_input_dir, dataset_name, "preprocessed")
    return test_data_dir, subject_ids


def get_transforms() -> Compose:
    """
    Get the transforms for the test data that they can be passed through the network
    Returns:
        The batchgenerators transformations which should be applied to the input
    """
    return Compose(
        [
            ResizeTransform(target_size=(128, 128), order=1),
            NumpyToTensor(),
        ]
    )


# TODO: probably generalize model here?
def load_model_from_checkpoint(checkpoint: Dict) -> UNet:
    """
    Load the model for the predictions from a checkpoint
    Args:
        checkpoint: The checkpoint to load the model from

    Returns:
        model [UNet]: The model for the predictions
    """
    hparams = checkpoint["hyper_parameters"]
    state_dict = OrderedDict()
    for k, v in checkpoint["state_dict"].items():
        state_dict[".".join(k.split(".")[1:])] = v

    model = UNet(hparams["num_classes"])
    model.load_state_dict(state_dict=state_dict)
    return model


def calculate_test_metrics(
    output_softmax: torch.Tensor, ground_truth: torch.Tensor
) -> Dict:
    """
    Calculate the metrics for evaluation
    Args:
        output [torch.Tensor]: The output of the network.
        output_softmax [torch.Tensor]: The output of the network after applying softmax.
        ground_truth [torch.Tensor]: The ground truth segmentation.

    Returns:
        metrics_dict [Dict]: A dictionary with the calculated metrics
    """
    dice_loss = SoftDiceLoss()
    nll_loss = torch.nn.NLLLoss()
    test_loss = dice_loss(output_softmax, ground_truth) + nll_loss(
        torch.log(output_softmax), ground_truth
    )
    test_dice = dice_score(output_softmax, ground_truth)
    metrics_dict = {"loss": test_loss.item(), "dice": test_dice.item()}
    return metrics_dict


def predict_cases(
    test_datacarrier: DataCarrier, data_samples: List[Dict], model: nn.Module
) -> DataCarrier:
    """
    Predict all test cases.
    Args:
        test_datacarrier: The datacarrier to save the data
        data_samples: The samples to predict
        model: The model used for prediction

    Returns:
        test_datacarrier: The datacarrier with the concatenated data
    """
    image_transforms = get_transforms()

    for sample in tqdm(data_samples):
        input = test_datacarrier.load_slice(sample)
        to_tensor = Compose([NumpyToTensor()])
        input_tensor = to_tensor(**input)
        input_transformed = image_transforms(**input)

        org_image_size = input["data"].shape
        model = model.double()
        output = model.forward(input_transformed["data"].double())
        resize = transforms.Resize(org_image_size[-2:])
        output_resized = resize(output)
        output_softmax = F.softmax(output_resized, dim=1)

        test_datacarrier.concat_data(batch=input_tensor, softmax_pred=output_softmax)
    return test_datacarrier


def calculate_metrics(test_datacarrier: DataCarrier) -> None:
    """
    Calculate metrics when all slices for all cases have been predicted
    Args:
        test_datacarrier: The datacarrier with the concatenated data
    """
    for key, value in test_datacarrier.data.items():
        softmax_pred = np.asarray(value["softmax_pred"])
        softmax_pred = np.moveaxis(softmax_pred, 0, 1)
        softmax_pred = torch.from_numpy(softmax_pred)
        softmax_pred = torch.unsqueeze(softmax_pred, 0)
        gt_seg = torch.from_numpy(np.asarray(value["seg"]))
        gt_seg = torch.unsqueeze(gt_seg, 0).type(torch.LongTensor)
        metrics_dict = calculate_test_metrics(softmax_pred, gt_seg)
        test_datacarrier.data[key]["metrics"] = metrics_dict


def save_results(test_datacarrier: DataCarrier, hparams: Dict, args: Namespace) -> None:
    """
    Save the results of the datacarrier to disc

    Args:
        test_datacarrier: The datacarrier which contains the data to save
        hparams: Dict with hyperparameters of training. Needed to infer the path where to store test results.
    """
    save_dir = args.save_dir if args.save_dir is not None else hparams["save_dir"]
    data_input_dir = (
        args.data_input_dir
        if args.data_input_dir is not None
        else hparams["data_input_dir"]
    )
    test_datacarrier.save_data(
        root_dir=save_dir,
        exp_name=hparams["exp_name"],
        version=hparams["version"],
        org_data_path=os.path.join(data_input_dir, hparams["dataset_name"], "imagesTr"),
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

    checkpoint = torch.load(args.checkpoint_path)
    hparams = checkpoint["hyper_parameters"]

    # No test data dir specified, so data should be in same input dir as training data and split should be specified
    if test_data_dir is None:
        test_data_dir, subject_ids = dir_and_subjects_from_train(hparams, args)

    test_datacarrier = DataCarrier()
    data_samples = test_datacarrier.get_data_samples(
        base_dir=test_data_dir,
        slice_offset=0,
        subject_ids=subject_ids,
        slice_axis=0,
    )

    model = load_model_from_checkpoint(checkpoint)
    test_datacarrier = predict_cases(test_datacarrier, data_samples, model)
    calculate_metrics(test_datacarrier)
    save_results(test_datacarrier, hparams, args)


if __name__ == "__main__":
    arguments = test_cli()
    run_test(arguments)
