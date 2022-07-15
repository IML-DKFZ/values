import os
from typing import Optional, Tuple, List
from argparse import Namespace, ArgumentParser

import hydra
import yaml

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.distributions as td
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from torchmetrics.functional.classification import dice

from loss_modules import SoftDiceLoss
from data_carrier_3D import DataCarrier3D


class LightningExperiment(pl.LightningModule):
    def __init__(
        self,
        hparams: DictConfig,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-6,
        nested_hparam_dict: Optional[dict] = None,
        aleatoric_loss: bool = False,
        n_aleatoric_samples: int = 10,
        *args,
        **kwargs
    ):
        """Experiment Class which handles the optimizer, training, validation & testing.
        Saves hparams as well as the nested_hparam_dict when instance is called in pl.Trainer.fit(model=unet_exp)

        Args:
            hparams ([dict/Namespace]): hparams
            learning_rate (float, optional): [learning rate for optimizer]. Defaults to 1e-4.
            weight_decay (float, optional): [weight decay on model]. Defaults to 1e-6.
            nested_hparam_dict (Optional[dict], optional): if dict -> saved in the experiment_directory. Defaults to None.
        """
        super(LightningExperiment, self).__init__()
        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)
        if "DATASET_LOCATION" in os.environ.keys():
            hparams.data_input_dir = os.environ["DATASET_LOCATION"]
        self.save_hyperparameters(OmegaConf.to_container(hparams))
        self.nested_hparam_dict = nested_hparam_dict

        if aleatoric_loss is not None:
            self.model = hydra.utils.instantiate(
                hparams.model, aleatoric_loss=aleatoric_loss
            )
        else:
            self.model = hydra.utils.instantiate(hparams.model)
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate

        self.aleatoric_loss = aleatoric_loss
        self.n_aleatoric_samples = n_aleatoric_samples
        self.dice_loss = SoftDiceLoss()
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.nll_loss = torch.nn.NLLLoss()

        self.val_loss_avg = 0.0
        self.val_avg_acc = 0.0
        self.test_datacarrier = DataCarrier3D()

    def configure_optimizers(self) -> Tuple[List[optim.Adam], List[dict]]:
        """Define the optimizers and learning rate schedulers. Adam is used as optimizer.

        Returns:
            optimizer [List[optim.Adam]]: The optimizer which is used in training (Adam)
            scheduler [dict]: The learning rate scheduler
        """
        optimizer = optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        # scheduler dictionary which defines scheduler and how it is used in the training loop
        scheduler = {
            "scheduler": lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer, patience=10
            ),
            "monitor": "validation/val_loss",
            "interval": "epoch",
            "frequency": 1,
        }
        return [optimizer], [scheduler]

    def on_fit_start(self):
        """Called when fit begins
        Be careful: on_fit_start is executed before the train_loop as well as the test_loop v 1.0.3
        Logs the hyperparameters.
        """

        # set placeholders for the metrics according to the stage of the trainer
        if self.trainer.testing is False:
            metric_placeholder = {
                "validation/val_loss": 0.0,
                "validation/val_dice": 0.0,
            }
        else:
            metric_placeholder = {"test/test_loss": 0.0, "test/test_dice": 0.0}

        self.hparams.version = self.logger.version
        # Save nested_hparam_dict if available
        if self.nested_hparam_dict is not None:

            with open(
                os.path.join(self.logger.experiment.log_dir, "hparams_sub_nested.yml"),
                "w",
            ) as file:
                yaml.dump(self.nested_hparam_dict, file, default_flow_style=False)

            sub_hparams = dict()
            for subdict in self.nested_hparam_dict.values():
                sub_hparams.update(subdict)
            sub_hparams = Namespace(**sub_hparams)
            self.logger.log_hyperparams(sub_hparams, metrics=metric_placeholder)
        else:
            self.logger.log_hyperparams(
                Namespace(**self.hparams), metrics=metric_placeholder
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor | td.LowRankMultivariateNormal:
        """Forward pass through the network

        Args:
            x: The input batch

        Returns:
            [torch.Tensor]: The result of the V-Net
        """
        return self.model(x)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Perform a training step, i.e. pass a batch to the network and calculate the loss.

        Args:
            batch (dict): The training batch
            batch_idx (int): The index of the current batch

        Returns:
            loss [torch.Tensor]: The computed loss
        """
        target = batch["seg"].long().squeeze()
        if self.aleatoric_loss is None:
            distribution = self.forward(batch["data"])
            samples = distribution.rsample([self.n_aleatoric_samples])
            samples = samples.view(
                [
                    self.n_aleatoric_samples,
                    batch["data"].size()[0],
                    self.model.num_classes,
                    *batch["data"].size()[-3:],
                ]
            )
            loss = torch.zeros([self.n_aleatoric_samples])
            for idx, sample in enumerate(samples):
                # loss[idx] = self.dice_loss(sample, target) + self.ce_loss(
                #     sample, target
                # )
                loss[idx] = self.ce_loss(sample, target)
            loss = torch.mean(loss)
        elif self.aleatoric_loss:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            mu, s = self.forward(batch["data"])
            sigma = torch.exp(s / 2)
            all_samples = torch.zeros(
                (self.n_aleatoric_samples, *mu.size()), device=device
            )
            for t in range(self.n_aleatoric_samples):
                epsilon = torch.randn(s.size(), device=device)
                sample = mu + sigma * epsilon
                log_sample_prob = F.log_softmax(sample)
                all_samples[t] = log_sample_prob
            log_sample_avg = torch.logsumexp(all_samples, 0) - torch.log(
                torch.tensor(self.n_aleatoric_samples)
            )
            loss = self.dice_loss(torch.exp(log_sample_avg), target) + self.nll_loss(
                log_sample_avg, target
            )
        else:
            output = self.forward(batch["data"])
            output_softmax = F.softmax(output, dim=1)

            loss = self.dice_loss(output_softmax, target) + self.ce_loss(output, target)
        self.log(
            "training/train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
            batch_size=self.hparams.batch_size,
        )
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Perform a validation step, i.e.pass a validation batch through the network, visualize the results in logging
        and calculate loss and dice score for logging

        Args:
            batch (dict): The validation batch
            batch_idx (int): The index of the current batch

        Returns:
            val_loss [torch.Tensor]: The computed loss
        """
        target = batch["seg"].long().squeeze()
        if len(target.size()) == 3:
            target = target.unsqueeze(0)

        if self.aleatoric_loss is None:
            distribution = self.forward(batch["data"])
            samples = distribution.rsample([self.n_aleatoric_samples])
            samples = samples.view(
                [
                    self.n_aleatoric_samples,
                    batch["data"].size()[0],
                    self.model.num_classes,
                    *batch["data"].size()[-3:],
                ]
            )
            val_loss = torch.zeros([self.n_aleatoric_samples])
            val_dice = torch.zeros([self.n_aleatoric_samples])
            for idx, sample in enumerate(samples):
                # val_loss[idx] = self.dice_loss(sample, target) + self.ce_loss(
                #     sample, target
                # )
                val_loss[idx] = self.ce_loss(sample, target)
                val_dice[idx] = dice(sample, target, ignore_index=0)
            val_loss = torch.mean(val_loss)
            val_dice = torch.mean(val_dice)
        elif self.aleatoric_loss:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            mu, s = self.forward(batch["data"])
            sigma = torch.exp(s / 2)
            all_samples = torch.zeros(
                (self.n_aleatoric_samples, *mu.size()), device=device
            )
            for t in range(self.n_aleatoric_samples):
                epsilon = torch.randn(s.size(), device=device)
                sample = mu + sigma * epsilon
                log_sample_prob = F.log_softmax(sample)
                all_samples[t] = log_sample_prob
            log_sample_avg = torch.logsumexp(all_samples, 0) - torch.log(
                torch.tensor(self.n_aleatoric_samples)
            )
            val_loss = self.dice_loss(
                torch.exp(log_sample_avg), target
            ) + self.nll_loss(log_sample_avg, target)
            val_dice = dice(torch.exp(log_sample_avg), target, ignore_index=0)
        else:
            output = self.forward(batch["data"].float())
            output_softmax = F.softmax(output, dim=1)
            val_loss = self.dice_loss(output_softmax, target) + self.ce_loss(
                output, target
            )
            val_dice = dice(output_softmax, target, ignore_index=0)

        # Visualization of Segmentations
        # TODO: Visualization for 3D?
        # if batch_idx == 1:
        #     self.predicted_segmentation_val = torch.argmax(output, dim=1, keepdim=True)
        #     self.target_segmentation_val = batch["seg"].long()
        #     grid = torchvision.utils.make_grid(self.predicted_segmentation_val)
        #     self.logger.experiment.add_image(
        #         "validation/Val_Predicted_Segmentations", grid, self.current_epoch
        #     )
        #     grid = torchvision.utils.make_grid(self.target_segmentation_val)
        #     self.logger.experiment.add_image(
        #         "validation/Val_Target_Segmentations", grid, self.current_epoch
        #     )

        log = {"validation/val_loss": val_loss, "validation/val_dice": val_dice}
        self.log_dict(log, prog_bar=False, on_step=False, on_epoch=True, logger=True)

        return val_loss

    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Perform a test step, i.e.pass a test batch through the network, calculate loss and dice score for logging
        and visualize the results in logging

        Args:
            batch (dict): The test batch
            batch_idx (int): The index of the current batch

        Returns:
            test_loss [torch.Tensor]: The computed loss
        """
        output = self.forward(batch["data"].float())
        output_softmax = F.softmax(output, dim=1)

        target = batch["seg"].long().squeeze()

        test_loss = self.dice_loss(output_softmax, target) + self.ce_loss(
            output, target
        )
        test_dice = dice(output_softmax, target, ignore_index=0)
        self.test_datacarrier.concat_data(batch=batch, softmax_pred=output_softmax)

        log = {"test/test_loss": test_loss, "test/test_dice": test_dice}
        self.log_dict(log, prog_bar=False, on_step=False, on_epoch=True, logger=True)
        return test_loss

    def on_test_end(self) -> None:
        self.test_datacarrier.save_data(
            root_dir=self.hparams.save_dir,
            exp_name=self.hparams.exp_name,
            version=self.logger.version,
        )

    @staticmethod
    def add_module_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """Add arguments to parser that are specific for experiment module (learning rate, weight decay and seed)

        Args:
            parent_parser (ArgumentParser): The parser to add the arguments

        Returns:
            parser [ArgumentParser]: The parent parser with the appended module specific arguments
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--learning_rate",
            type=float,
            help="Learning rate.",
            default=1e-4,
        )
        parser.add_argument(
            "--weight_decay",
            type=float,
            help="Weight decay value for optimizer.",
            default=1e-6,
        )
        parser.add_argument(
            "--seed", type=int, help="Random seed for training", default=123
        )
        return parser


if __name__ == "__main__":
    trainer = pl.Trainer()
    trainer.test()
    pass
