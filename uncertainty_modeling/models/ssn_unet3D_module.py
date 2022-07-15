import numpy as np

from uncertainty_modeling.models.unet3D_module import UNet3D
import torch
import torch.nn as nn
import torch.distributions as td


class SsnUNet3D(UNet3D):
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        initial_filter_size: int = 8,
        kernel_size: int = 3,
        do_instancenorm: bool = True,
        do_dropout: bool = False,
        rank: int = 10,
        epsilon: float = 1e-5,
    ):
        super().__init__(
            num_classes * 2 + num_classes * rank,
            in_channels,
            initial_filter_size,
            kernel_size,
            do_instancenorm,
            do_dropout,
            aleatoric_loss=False,
        )
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.rank = rank
        self.mean_conv = nn.Conv3d(num_classes, num_classes, kernel_size=1)
        self.log_cov_diag_conv = nn.Conv3d(num_classes, num_classes, kernel_size=1)
        self.cov_factor_conv = nn.Conv3d(num_classes, num_classes * rank, kernel_size=1)

    def forward(self, x: torch.Tensor, enable_concat: bool = True):
        logits = super().forward(x, enable_concat)
        batch_size = logits.shape[0]

        mean = logits[:, : self.num_classes]
        cov_diag = (
            logits[:, self.num_classes : 2 * self.num_classes].exp() + self.epsilon
        )
        mean = mean.view((batch_size, -1))
        cov_diag = cov_diag.view((batch_size, -1))

        cov_factor = logits[:, 2 * self.num_classes :]
        cov_factor = cov_factor.view((batch_size, self.rank, self.num_classes, -1))
        cov_factor = cov_factor.flatten(2, 3)
        cov_factor = cov_factor.transpose(1, 2)

        x_flat = torch.flatten(x, 1)
        mode = torch.mode(x_flat)
        masks = []
        zeros = torch.zeros_like(x_flat[0])
        ones = torch.ones_like(x_flat[0])
        for idx, value in enumerate(mode.values):
            mask = torch.where(x_flat[idx] == value, zeros, ones)
            masks.append(mask.repeat(self.num_classes))
        mask = torch.stack(masks)
        cov_factor = cov_factor * mask.unsqueeze(-1)
        cov_diag = cov_diag * mask + self.epsilon
        cov_factor = torch.tanh(cov_factor) * 5
        # print("Covariance")
        # print(cov_factor.max())
        # print(cov_factor.min())
        #
        # print("Location")
        # print(mean.max())
        # print(mean.min())
        try:
            distribution = td.LowRankMultivariateNormal(
                loc=mean, cov_factor=cov_factor, cov_diag=cov_diag
            )
        except:
            print(
                "Covariance became not invertible using independent normals for this batch!"
            )
            distribution = td.Independent(
                td.Normal(loc=mean, scale=torch.sqrt(cov_diag)), 1
            )

        return distribution
