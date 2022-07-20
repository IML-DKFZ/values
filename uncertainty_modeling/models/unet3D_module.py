from typing import Tuple

import torch
import torch.nn as nn


class UNet3D(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        initial_filter_size: int = 8,
        kernel_size: int = 3,
        do_instancenorm: bool = True,
        do_dropout: bool = False,
        aleatoric_loss: bool = False,
    ):
        """Unet (torch.nn.Module) for 3D input segmentation

        Args:
            num_classes ([int]): [Number of output classes for segmentation]
            in_channels (int, optional): [num in_channels]. Defaults to 1.
            initial_filter_size (int, optional): [number of filters for factor]. Defaults to 64.
            kernel_size (int, optional): [size of kernels]. Defaults to 3.
            do_instancenorm (bool, optional): [True: use instancenorm]. Defaults to True.
        """
        super().__init__()
        self.num_classes = num_classes
        self.aleatoric_loss = aleatoric_loss

        if do_dropout:
            self.dropout_prob = 0.5
        else:
            self.dropout_prob = 0.0

        self.contr_1_1 = self.contract(
            in_channels,
            initial_filter_size,
            kernel_size,
            instancenorm=do_instancenorm,
            dropout_prob=self.dropout_prob,
        )
        self.contr_1_2 = self.contract(
            initial_filter_size,
            initial_filter_size,
            kernel_size,
            instancenorm=do_instancenorm,
            dropout_prob=self.dropout_prob,
        )
        self.pool = nn.MaxPool3d(2, stride=2)

        self.contr_2_1 = self.contract(
            initial_filter_size,
            initial_filter_size * 2,
            kernel_size,
            instancenorm=do_instancenorm,
            dropout_prob=self.dropout_prob,
        )
        self.contr_2_2 = self.contract(
            initial_filter_size * 2,
            initial_filter_size * 2,
            kernel_size,
            instancenorm=do_instancenorm,
            dropout_prob=self.dropout_prob,
        )

        self.contr_3_1 = self.contract(
            initial_filter_size * 2,
            initial_filter_size * 2**2,
            kernel_size,
            instancenorm=do_instancenorm,
            dropout_prob=self.dropout_prob,
        )
        self.contr_3_2 = self.contract(
            initial_filter_size * 2**2,
            initial_filter_size * 2**2,
            kernel_size,
            instancenorm=do_instancenorm,
            dropout_prob=self.dropout_prob,
        )

        self.contr_4_1 = self.contract(
            initial_filter_size * 2**2,
            initial_filter_size * 2**3,
            kernel_size,
            instancenorm=do_instancenorm,
            dropout_prob=self.dropout_prob,
        )
        self.contr_4_2 = self.contract(
            initial_filter_size * 2**3,
            initial_filter_size * 2**3,
            kernel_size,
            instancenorm=do_instancenorm,
            dropout_prob=self.dropout_prob,
        )

        if do_dropout:
            self.center = nn.Sequential(
                nn.Conv3d(
                    initial_filter_size * 2**3,
                    initial_filter_size * 2**4,
                    3,
                    padding=1,
                ),
                nn.ReLU(inplace=True),
                nn.Conv3d(
                    initial_filter_size * 2**4,
                    initial_filter_size * 2**4,
                    3,
                    padding=1,
                ),
                nn.ReLU(inplace=True),
                nn.ConvTranspose3d(
                    initial_filter_size * 2**4,
                    initial_filter_size * 2**3,
                    2,
                    stride=2,
                ),
                nn.ReLU(inplace=True),
                nn.Dropout(p=self.dropout_prob),
            )
        else:
            self.center = nn.Sequential(
                nn.Conv3d(
                    initial_filter_size * 2**3,
                    initial_filter_size * 2**4,
                    3,
                    padding=1,
                ),
                nn.ReLU(inplace=True),
                nn.Conv3d(
                    initial_filter_size * 2**4,
                    initial_filter_size * 2**4,
                    3,
                    padding=1,
                ),
                nn.ReLU(inplace=True),
                nn.ConvTranspose3d(
                    initial_filter_size * 2**4,
                    initial_filter_size * 2**3,
                    2,
                    stride=2,
                ),
                nn.ReLU(inplace=True),
            )

        self.expand_4_1 = self.expand(
            initial_filter_size * 2**4,
            initial_filter_size * 2**3,
            dropout_prob=self.dropout_prob,
        )
        self.expand_4_2 = self.expand(
            initial_filter_size * 2**3,
            initial_filter_size * 2**3,
            dropout_prob=self.dropout_prob,
        )
        self.upscale4 = nn.ConvTranspose3d(
            initial_filter_size * 2**3,
            initial_filter_size * 2**2,
            kernel_size=2,
            stride=2,
        )

        self.expand_3_1 = self.expand(
            initial_filter_size * 2**3,
            initial_filter_size * 2**2,
            dropout_prob=self.dropout_prob,
        )
        self.expand_3_2 = self.expand(
            initial_filter_size * 2**2,
            initial_filter_size * 2**2,
            dropout_prob=self.dropout_prob,
        )
        self.upscale3 = nn.ConvTranspose3d(
            initial_filter_size * 2**2, initial_filter_size * 2, 2, stride=2
        )

        self.expand_2_1 = self.expand(
            initial_filter_size * 2**2,
            initial_filter_size * 2,
            dropout_prob=self.dropout_prob,
        )
        self.expand_2_2 = self.expand(
            initial_filter_size * 2,
            initial_filter_size * 2,
            dropout_prob=self.dropout_prob,
        )
        self.upscale2 = nn.ConvTranspose3d(
            initial_filter_size * 2, initial_filter_size, 2, stride=2
        )

        self.expand_1_1 = self.expand(
            initial_filter_size * 2, initial_filter_size, dropout_prob=self.dropout_prob
        )
        self.expand_1_2 = self.expand(
            initial_filter_size, initial_filter_size, dropout_prob=self.dropout_prob
        )
        # Output layer for segmentation, kernel size for final layer = 1, see paper
        self.final = nn.Conv3d(initial_filter_size, num_classes, kernel_size=1)
        if self.aleatoric_loss:
            self.final_aleatoric = nn.Conv3d(
                initial_filter_size, num_classes * 2, kernel_size=1
            )

        self.softmax = torch.nn.Softmax()

        # Output layer for "autoencoder-mode"
        self.output_reconstruction_map = nn.Conv3d(
            initial_filter_size, out_channels=1, kernel_size=1
        )

    @staticmethod
    def contract(
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        instancenorm: bool = True,
        dropout_prob: float = 0.0,
    ) -> nn.Sequential:
        """One contraction in U-net. Consists of one convolution and ReLU

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int, optional): Kernel size for the convolution
            instancenorm (bool, optional): Whether to apply instance norm

        Returns:
            layer [nn.Sequential]: sequential container with all layers of the contraction
        """
        if instancenorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, padding=1),
                nn.InstanceNorm3d(out_channels),
                nn.LeakyReLU(inplace=True),
                nn.Dropout(p=dropout_prob),
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.Dropout(p=dropout_prob),
            )
        return layer

    @staticmethod
    def expand(
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout_prob: float = 0.0,
    ) -> nn.Sequential:
        """One expansion in U-net. Consists of one convolution and ReLU

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int, optional): Kernel size for the convolution

        Returns:
            layer [nn.Sequential]: sequential container with all layers of the contraction
        """
        layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=dropout_prob),
        )
        return layer

    @staticmethod
    def center_crop(
        image: torch.Tensor, target_depth: int, target_height: int, target_width: int
    ) -> torch.Tensor:
        """Perform center crop for concatenation in expansion path

        Args:
            layer (torch.Tensor): The input batch to crop
            target_width (int): The desired width after cropping
            target_height (int): The desired height after cropping

        Returns:
            [torch.Tensor]: The cropped batch
        """
        batch_size, n_channels, image_depth, image_height, image_width = image.size()
        xyz1 = (image_depth - target_depth) // 2
        xyz2 = (image_height - target_height) // 2
        xyz3 = (image_width - target_width) // 2
        return image[
            :,
            :,
            xyz1 : (xyz1 + target_depth),
            xyz2 : (xyz2 + target_height),
            xyz3 : (xyz3 + target_width),
        ]

    def forward(
        self, x: torch.Tensor, enable_concat: bool = True, last_layer: bool = True
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network

        Args:
            x (torch.Tensor): input batch
            enable_concat: Whether to use skip connections

        Returns:
            output (torch.Tensor): The result of the network
        """
        concat_weight = 1
        if not enable_concat:
            concat_weight = 0

        # Encoder
        contr_1 = self.contr_1_2(self.contr_1_1(x))
        pool = self.pool(contr_1)

        contr_2 = self.contr_2_2(self.contr_2_1(pool))
        pool = self.pool(contr_2)

        contr_3 = self.contr_3_2(self.contr_3_1(pool))
        pool = self.pool(contr_3)

        contr_4 = self.contr_4_2(self.contr_4_1(pool))
        pool = self.pool(contr_4)

        # Bottleneck
        center = self.center(pool)

        # Decoder
        crop = self.center_crop(
            contr_4, center.size()[2], center.size()[3], center.size()[4]
        )
        concat = torch.cat([center, crop * concat_weight], 1)

        expand = self.expand_4_2(self.expand_4_1(concat))
        upscale = self.upscale4(expand)

        crop = self.center_crop(
            contr_3, upscale.size()[2], upscale.size()[3], upscale.size()[4]
        )
        concat = torch.cat([upscale, crop * concat_weight], 1)

        expand = self.expand_3_2(self.expand_3_1(concat))
        upscale = self.upscale3(expand)

        crop = self.center_crop(
            contr_2, upscale.size()[2], upscale.size()[3], upscale.size()[4]
        )
        concat = torch.cat([upscale, crop * concat_weight], 1)

        expand = self.expand_2_2(self.expand_2_1(concat))
        upscale = self.upscale2(expand)

        crop = self.center_crop(
            contr_1, upscale.size()[2], upscale.size()[3], upscale.size()[4]
        )
        concat = torch.cat([upscale, crop * concat_weight], 1)

        expand = self.expand_1_2(self.expand_1_1(concat))

        if not last_layer:
            return expand

        if enable_concat:
            if not self.aleatoric_loss:
                output = self.final(expand)
            else:
                output = self.final_aleatoric(expand)
                mu, s = output.split(self.num_classes, 1)
                return mu, s
        else:
            output = self.output_reconstruction_map(expand)

        return output
