import json
import os
from argparse import Namespace

import albumentations
import cv2
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torchmetrics.functional import dice
import torch.nn.functional as F
from tqdm import tqdm

from uncertainty_modeling.main import set_seed
from uncertainty_modeling.test_3D import (
    test_cli,
    load_models_from_checkpoint,
    calculate_ged,
    calculate_uncertainty,
    calculate_one_minus_msr,
)
import uncertainty_modeling.data.cityscapes_labels as cs_labels


class Tester:
    def __init__(self, args: Namespace):
        self.all_checkpoints = self.get_checkpoints(args.checkpoint_paths)
        hparams = self.all_checkpoints[0]["hyper_parameters"]
        set_seed(hparams["seed"])
        self.ignore_index = hparams["datamodule"]["ignore_index"]
        self.test_batch_size = args.test_batch_size
        self.tta = args.tta
        self.test_dataloader = self.get_test_dataloader(args, hparams)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = load_models_from_checkpoint(
            self.all_checkpoints, device=self.device
        )
        self.n_pred = args.n_pred
        self.results_dict = {}
        self.save_root_dir = (
            args.save_dir if args.save_dir is not None else hparams["save_dir"]
        )
        self.exp_name = hparams["exp_name"] if args.exp_name is None else args.exp_name
        self.version = str(hparams["version"])
        self.test_split = args.test_split
        self.create_save_dirs()

    @staticmethod
    def get_checkpoints(checkpoint_paths):
        all_checkpoints = []
        for checkpoint_path in checkpoint_paths:
            checkpoint = torch.load(checkpoint_path)
            checkpoint["hyper_parameters"]["MODEL"]["PRETRAINED"] = False
            conf = OmegaConf.create(checkpoint["hyper_parameters"])
            resolved = OmegaConf.to_container(conf, resolve=True)
            checkpoint["hyper_parameters"] = resolved
            all_checkpoints.append(checkpoint)
        return all_checkpoints

    @staticmethod
    def set_n_reference_samples(hparams, n_reference_samples):
        label_switch_index = [
            i
            for i, aug in enumerate(
                hparams["AUGMENTATIONS"]["TEST"][0]["Compose"]["transforms"]
            )
            if "StochasticLabelSwitches" in aug
        ][0]
        hparams["AUGMENTATIONS"]["TEST"][0]["Compose"]["transforms"][
            label_switch_index
        ]["StochasticLabelSwitches"]["n_reference_samples"] = n_reference_samples
        return hparams

    def create_save_dirs(self):
        self.save_dir = os.path.join(
            self.save_root_dir,
            self.exp_name,
            "test_results",
            self.version,
            self.test_split,
        )
        print(f"saving to results dir {self.save_dir}")
        self.save_pred_dir = os.path.join(self.save_dir, "pred_seg")
        self.save_pred_prob_dir = os.path.join(self.save_dir, "pred_prob")
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.save_pred_dir, exist_ok=True)
        # os.makedirs(self.save_pred_prob_dir, exist_ok=True)
        return

    def get_test_dataloader(self, args: Namespace, hparams):
        data_input_dir = (
            args.data_input_dir
            if args.data_input_dir is not None
            else hparams["data_input_dir"]
        )
        if args.data_input_dir is not None:
            hparams["datamodule"]["dataset"]["splits_path"] = hparams["datamodule"][
                "dataset"
            ]["splits_path"].replace(hparams["data_input_dir"], args.data_input_dir)
        hparams = self.set_n_reference_samples(hparams, args.n_reference_samples)
        if self.test_batch_size:
            hparams["datamodule"]["val_batch_size"] = self.test_batch_size
        dm = hydra.utils.instantiate(
            hparams["datamodule"],
            data_input_dir=data_input_dir,
            augmentations=hparams["AUGMENTATIONS"],
            seed=hparams["seed"],
            test_split=args.test_split,
            tta=self.tta,
            _recursive_=False,
        )
        dm.setup("test")
        return dm.test_dataloader()

    def save_prediction(self, image_id, image_preds, mean_pred, ignore_index_map):
        multiple_preds = False
        if image_preds.shape[0] > 1:
            image_preds_mean = torch.cat([mean_pred.unsqueeze(0), image_preds], dim=0)
            multiple_preds = True
        else:
            image_preds_mean = image_preds
        for output_idx, output in enumerate(image_preds_mean):
            output = torch.moveaxis(output, 0, -1)
            # output_softmax_np = output.detach().cpu().numpy()
            output = torch.argmax(output, dim=-1, keepdim=True)
            output_np = output.detach().long().cpu().numpy().astype(np.uint8)
            output_np_color = np.zeros((*output_np.shape[:-1], 3), dtype=np.uint8)
            output_np[ignore_index_map.astype(bool)] = cs_labels.name2trainId[
                "unlabeled"
            ]
            for k, v in cs_labels.trainId2color.items():
                output_np_color[(output_np == k).squeeze(-1), :] = v
            if not multiple_preds:
                output_idx += 1
            img_name = (
                f"{image_id}_mean"
                if output_idx == 0 and multiple_preds
                else f"{image_id}_{str(output_idx).zfill(2)}"
            )
            # np.savez_compressed(
            #     os.path.join(self.save_pred_prob_dir, f"{img_name}.npz"),
            #     pred_prob=output_softmax_np,
            # )
            output_np_color = cv2.cvtColor(output_np_color, cv2.COLOR_BGR2RGB)
            cv2.imwrite(
                os.path.join(self.save_pred_dir, f"{img_name}.png"), output_np_color
            )
        return

    def save_uncertainty(self, image_id, uncertainty_dict):
        for unc_type, unc_map in uncertainty_dict.items():
            unc_dir = os.path.join(self.save_dir, unc_type)
            os.makedirs(unc_dir, exist_ok=True)
            # TODO: Choose good file format that can handle floating point numbers
            # unc_map_np = (unc_map.detach().cpu().numpy() * 255).astype(np.uint8)
            unc_map_np = unc_map.detach().cpu().numpy()
            cv2.imwrite(os.path.join(unc_dir, f"{image_id}.tif"), unc_map_np)
            # save(unc_map_np, os.path.join(unc_dir, f"{image_id}.nii.gz"))

    def calculate_test_metrics(self, output_softmax, ground_truth):
        metrics_dict = {}
        output_softmax = torch.unsqueeze(output_softmax, 0).type(torch.FloatTensor)
        all_test_dice = []
        for rater in ground_truth:
            rater = torch.unsqueeze(rater, 0)
            test_dice = dice(
                output_softmax, rater, ignore_index=output_softmax.shape[1] - 1
            )
            all_test_dice.append(test_dice.item())
        metrics_dict["dice"] = np.mean(np.array(all_test_dice))
        # self.results_dict[image_id]["metrics"].update(metrics_dict)
        return metrics_dict

    def process_image_prediction(
        self, all_preds, image_idx, image_preds, ignore_index_map
    ):
        image_id = all_preds["image_id"][image_idx]
        mean_softmax_pred = torch.mean(image_preds, dim=0)
        self.results_dict[image_id] = {"dataset": all_preds["dataset"][image_idx]}
        self.results_dict[image_id]["metrics"] = {}
        self.results_dict[image_id]["metrics"].update(
            self.calculate_test_metrics(mean_softmax_pred, all_preds["gt"][image_idx])
        )
        self.results_dict[image_id]["metrics"].update(
            calculate_ged(
                image_preds,
                all_preds["gt"][image_idx],
                ignore_index=image_preds.shape[1] - 1,
            )
        )
        if image_preds.shape[0] > 1:
            uncertainty_dict = calculate_uncertainty(image_preds)
        else:
            uncertainty_dict = calculate_one_minus_msr(image_preds.squeeze(0))
        ignore_index_map_image = ignore_index_map[image_idx][0].unsqueeze(-1)
        self.save_prediction(
            image_id,
            image_preds,
            mean_softmax_pred,
            ignore_index_map_image.detach().long().cpu().numpy().astype(np.uint8),
        )
        self.save_uncertainty(image_id, uncertainty_dict)

    def process_output(self, all_preds, is_ssn):
        pred_shape = all_preds["softmax_pred"].shape
        # The extra dimension is added to enable that torchmetrics can deal with ignore index outside of softmax dimensions
        extra_dimension = torch.zeros(
            pred_shape[0],
            pred_shape[1],
            1,
            pred_shape[3],
            pred_shape[4],
            device=self.device,
        )
        all_preds["softmax_pred"] = torch.cat(
            (all_preds["softmax_pred"], extra_dimension), dim=2
        )
        ignore_index_map = all_preds["gt"] == self.ignore_index
        all_preds["gt"][all_preds["gt"] == self.ignore_index] = (
            all_preds["softmax_pred"].shape[2] - 1
        )
        image_predictions = [
            all_preds["softmax_pred"][:, i, :, :]
            for i in range(all_preds["softmax_pred"].shape[1])
        ]
        for image_idx, image_preds in enumerate(image_predictions):
            image_id = all_preds["image_id"][image_idx]
            mean_softmax_pred = torch.mean(image_preds, dim=0)
            self.results_dict[image_id] = {"dataset": all_preds["dataset"][image_idx]}
            self.results_dict[image_id]["metrics"] = {}
            self.results_dict[image_id]["metrics"].update(
                self.calculate_test_metrics(
                    mean_softmax_pred, all_preds["gt"][image_idx]
                )
            )
            self.results_dict[image_id]["metrics"].update(
                calculate_ged(
                    image_preds,
                    all_preds["gt"][image_idx].to(self.device),
                    ignore_index=image_preds.shape[1] - 1,
                    ged_only=True,
                )
            )
            if image_preds.shape[0] > 1:
                uncertainty_dict = calculate_uncertainty(image_preds, ssn=is_ssn)
            else:
                uncertainty_dict = calculate_one_minus_msr(image_preds.squeeze(0))
            ignore_index_map_image = ignore_index_map[image_idx][0].unsqueeze(-1)
            self.save_prediction(
                image_id,
                image_preds,
                mean_softmax_pred,
                ignore_index_map_image.detach().long().cpu().numpy().astype(np.uint8),
            )
            self.save_uncertainty(image_id, uncertainty_dict)

    def save_results_dict(self):
        filename = os.path.join(self.save_dir, "metrics.json")
        mean_metrics_dict = {}
        for image_id, value in self.results_dict.items():
            for metric, score in value["metrics"].items():
                if metric not in mean_metrics_dict:
                    mean_metrics_dict[metric] = []
                mean_metrics_dict[metric].append(score)
        self.results_dict["mean"] = {}
        self.results_dict["mean"]["metrics"] = {}
        for metric, scores in mean_metrics_dict.items():
            self.results_dict["mean"]["metrics"][metric] = np.asarray(scores).mean()
        with open(filename, "w") as f:
            json.dump(self.results_dict, f, indent=2)

    def predict_cases(self):
        for batch in tqdm(self.test_dataloader):
            # dataloader_iterator = iter(self.test_dataloader)
            # for i in tqdm(range(2)):
            #     batch = next(dataloader_iterator)
            all_preds = {
                "softmax_pred": [],
                "image_id": batch["image_id"],
                "gt": batch["seg"],
                "dataset": batch["dataset"],
            }
            for model in self.models:
                if model.ssn:
                    distribution = model.forward(batch["data"].to(self.device))
                    output_samples = distribution.sample([self.n_pred])
                    output_samples = output_samples.view(
                        [
                            self.n_pred,
                            batch["data"].size()[0],
                            model.num_classes,
                            *batch["data"].size()[2:],
                        ]
                    )
                    for output_sample in output_samples:
                        output_softmax = F.softmax(output_sample, dim=1)
                        all_preds["softmax_pred"].append(output_softmax)
                elif self.tta:
                    for index, image in enumerate(batch["data"]):
                        output = model.forward(image.to(self.device))
                        output_softmax = F.softmax(output, dim=1)  # .to("cpu")
                        if any(
                            "HorizontalFlip" in sl for sl in batch["transforms"][index]
                        ):
                            # all_preds["softmax_pred"].append(output_softmax)
                            all_preds["softmax_pred"].append(
                                torch.flip(output_softmax, [-1])
                            )
                        else:
                            all_preds["softmax_pred"].append(output_softmax)
                else:
                    for pred in range(self.n_pred):
                        output = model.forward(batch["data"].to(self.device))
                        output_softmax = F.softmax(output, dim=1)  # .to("cpu")
                        all_preds["softmax_pred"].append(output_softmax)
            all_preds["softmax_pred"] = torch.stack(all_preds["softmax_pred"])
            self.process_output(all_preds, is_ssn=self.models[0].ssn)
        self.save_results_dict()


def run_test(args: Namespace) -> None:
    """
    Run test and save the results in the end
    Args:
        args: Arguments for testing, including checkpoint_path, test_data_dir and subject_ids.
              test_data_dir and subject_ids might be None.
    """
    torch.set_grad_enabled(False)
    tester = Tester(args)
    tester.predict_cases()


if __name__ == "__main__":
    arguments = test_cli()
    run_test(arguments)
