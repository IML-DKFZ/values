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
        self.mean_conv = nn.Conv3d(initial_filter_size, num_classes, kernel_size=1)
        self.log_cov_diag_conv = nn.Conv3d(
            initial_filter_size, num_classes, kernel_size=1
        )
        self.cov_factor_conv = nn.Conv3d(
            initial_filter_size, num_classes * rank, kernel_size=1
        )

    def forward(
        self, x: torch.Tensor, enable_concat: bool = True, mean_only: bool = False
    ):
        logits = super().forward(x, enable_concat, last_layer=False)
        batch_size = logits.shape[0]

        mean = self.mean_conv(logits)
        mean = mean.view((batch_size, -1))
        cov_diag = self.log_cov_diag_conv(logits).exp() + self.epsilon
        cov_diag = cov_diag.view((batch_size, -1))

        if mean_only:
            cov_factor = torch.zeros([*cov_diag.shape, self.rank])
        else:
            cov_factor = self.cov_factor_conv(logits)
            cov_factor = cov_factor.view((batch_size, self.rank, self.num_classes, -1))
            cov_factor = cov_factor.flatten(2, 3)
            cov_factor = cov_factor.transpose(1, 2)

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
