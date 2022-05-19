import os
from typing import Optional, Tuple, List
import torchvision
from argparse import Namespace, ArgumentParser
import yaml

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import pytorch_lightning as pl
from torchmetrics.functional.classification import dice_score

from unet_module import UNet
from loss_modules import SoftDiceLoss
from data_carrier import DataCarrier


class UNetExperiment(pl.LightningModule):
    def __init__(
        self,
        hparams: Namespace,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-6,
        nested_hparam_dict: Optional[dict] = None,
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
        super(UNetExperiment, self).__init__()
        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)
        if "DATASET_LOCATION" in os.environ.keys():
            hparams.data_input_dir = os.environ["DATASET_LOCATION"]
        self.save_hyperparameters(hparams)
        self.nested_hparam_dict = nested_hparam_dict

        self.unet = UNet(
            num_classes=hparams.num_classes,
            in_channels=1,
            initial_filter_size=64,
            kernel_size=3,
            do_instancenorm=True,
        )
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate

        self.dice_loss = SoftDiceLoss()
        self.ce_loss = torch.nn.CrossEntropyLoss()

        self.val_loss_avg = 0.0
        self.val_avg_acc = 0.0
        self.test_datacarrier = DataCarrier()

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
            self.logger.log_hyperparams(self.hparams, metrics=metric_placeholder)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network

        Args:
            x: The input batch

        Returns:
            [torch.Tensor]: The result of the U-Net
        """
        return self.unet(x)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Perform a training step, i.e. pass a batch to the network and calculate the loss.

        Args:
            batch (dict): The training batch
            batch_idx (int): The index of the current batch

        Returns:
            loss [torch.Tensor]: The computed loss
        """
        output = self.forward(batch["data"])
        output_softmax = F.softmax(output, dim=1)

        target = batch["seg"].long().squeeze()
        loss = self.dice_loss(output_softmax, target) + self.ce_loss(output, target)

        self.log(
            "training/train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
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
        output = self.forward(batch["data"].float())
        output_softmax = F.softmax(output, dim=1)

        target = batch["seg"].long().squeeze()

        # Visualization of Segmentations
        if batch_idx == 1:
            self.predicted_segmentation_val = torch.argmax(output, dim=1, keepdim=True)
            self.target_segmentation_val = batch["seg"].long()
            grid = torchvision.utils.make_grid(self.predicted_segmentation_val)
            self.logger.experiment.add_image(
                "validation/Val_Predicted_Segmentations", grid, self.current_epoch
            )
            grid = torchvision.utils.make_grid(self.target_segmentation_val)
            self.logger.experiment.add_image(
                "validation/Val_Target_Segmentations", grid, self.current_epoch
            )

        val_loss = self.dice_loss(output_softmax, target) + self.ce_loss(output, target)
        val_dice = dice_score(output_softmax, target)

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
        test_dice = dice_score(output_softmax, target)
        self.test_datacarrier.concat_data(batch=batch, softmax_pred=output_softmax)
        if batch_idx == 1:
            self.predicted_segmentation_test = torch.argmax(output, dim=1, keepdim=True)
            self.target_segmentation_test = batch["seg"].long()
            self.batch_data = batch["data"]
            grid = torchvision.utils.make_grid(self.batch_data)
            self.logger.experiment.add_image(
                "test/Test_Input_Batch", grid, self.current_epoch
            )
            grid = torchvision.utils.make_grid(self.predicted_segmentation_test)
            self.logger.experiment.add_image(
                "test/Test_Predicted_Segmentations", grid, self.current_epoch
            )
            grid = torchvision.utils.make_grid(self.target_segmentation_test)
            self.logger.experiment.add_image(
                "test/Test_Target_Segmentations", grid, self.current_epoch
            )

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
