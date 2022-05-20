import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        initial_filter_size: int = 64,
        kernel_size: int = 3,
        do_instancenorm: bool = True,
    ):
        """Unet (torch.nn.Module) for 2D input segmentation

        Args:
            num_classes ([int]): [Number of output classes for segmentation]
            in_channels (int, optional): [num in_channels]. Defaults to 1.
            initial_filter_size (int, optional): [number of filters for factor]. Defaults to 64.
            kernel_size (int, optional): [size of kernels]. Defaults to 3.
            do_instancenorm (bool, optional): [True: use instancenorm]. Defaults to True.
        """
        super().__init__()

        self.contr_1_1 = self.contract(
            in_channels, initial_filter_size, kernel_size, instancenorm=do_instancenorm
        )
        self.contr_1_2 = self.contract(
            initial_filter_size,
            initial_filter_size,
            kernel_size,
            instancenorm=do_instancenorm,
        )
        self.pool = nn.MaxPool2d(2, stride=2)

        self.contr_2_1 = self.contract(
            initial_filter_size,
            initial_filter_size * 2,
            kernel_size,
            instancenorm=do_instancenorm,
        )
        self.contr_2_2 = self.contract(
            initial_filter_size * 2,
            initial_filter_size * 2,
            kernel_size,
            instancenorm=do_instancenorm,
        )
        # self.pool2 = nn.MaxPool2d(2, stride=2)

        self.contr_3_1 = self.contract(
            initial_filter_size * 2,
            initial_filter_size * 2**2,
            kernel_size,
            instancenorm=do_instancenorm,
        )
        self.contr_3_2 = self.contract(
            initial_filter_size * 2**2,
            initial_filter_size * 2**2,
            kernel_size,
            instancenorm=do_instancenorm,
        )
        # self.pool3 = nn.MaxPool2d(2, stride=2)

        self.contr_4_1 = self.contract(
            initial_filter_size * 2**2,
            initial_filter_size * 2**3,
            kernel_size,
            instancenorm=do_instancenorm,
        )
        self.contr_4_2 = self.contract(
            initial_filter_size * 2**3,
            initial_filter_size * 2**3,
            kernel_size,
            instancenorm=do_instancenorm,
        )
        # self.pool4 = nn.MaxPool2d(2, stride=2)

        self.center = nn.Sequential(
            nn.Conv2d(
                initial_filter_size * 2**3, initial_filter_size * 2**4, 3, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                initial_filter_size * 2**4, initial_filter_size * 2**4, 3, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                initial_filter_size * 2**4, initial_filter_size * 2**3, 2, stride=2
            ),
            nn.ReLU(inplace=True),
        )

        self.expand_4_1 = self.expand(
            initial_filter_size * 2**4, initial_filter_size * 2**3
        )
        self.expand_4_2 = self.expand(
            initial_filter_size * 2**3, initial_filter_size * 2**3
        )
        self.upscale4 = nn.ConvTranspose2d(
            initial_filter_size * 2**3,
            initial_filter_size * 2**2,
            kernel_size=2,
            stride=2,
        )

        self.expand_3_1 = self.expand(
            initial_filter_size * 2**3, initial_filter_size * 2**2
        )
        self.expand_3_2 = self.expand(
            initial_filter_size * 2**2, initial_filter_size * 2**2
        )
        self.upscale3 = nn.ConvTranspose2d(
            initial_filter_size * 2**2, initial_filter_size * 2, 2, stride=2
        )

        self.expand_2_1 = self.expand(
            initial_filter_size * 2**2, initial_filter_size * 2
        )
        self.expand_2_2 = self.expand(initial_filter_size * 2, initial_filter_size * 2)
        self.upscale2 = nn.ConvTranspose2d(
            initial_filter_size * 2, initial_filter_size, 2, stride=2
        )

        self.expand_1_1 = self.expand(initial_filter_size * 2, initial_filter_size)
        self.expand_1_2 = self.expand(initial_filter_size, initial_filter_size)
        # Output layer for segmentation, kernel size for final layer = 1, see paper
        self.final = nn.Conv2d(initial_filter_size, num_classes, kernel_size=1)

        self.softmax = torch.nn.Softmax2d()

        # Output layer for "autoencoder-mode"
        self.output_reconstruction_map = nn.Conv2d(
            initial_filter_size, out_channels=1, kernel_size=1
        )

    @staticmethod
    def contract(
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        instancenorm: bool = True,
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
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.LeakyReLU(inplace=True),
            )
        else:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
                nn.LeakyReLU(inplace=True),
            )
        return layer

    @staticmethod
    def expand(
        in_channels: int, out_channels: int, kernel_size: int = 3
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
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        return layer

    @staticmethod
    def center_crop(
        layer: torch.Tensor, target_width: int, target_height: int
    ) -> torch.Tensor:
        """Perform center crop for concatenation in expansion path

        Args:
            layer (torch.Tensor): The input batch to crop
            target_width (int): The desired width after cropping
            target_height (int): The desired height after cropping

        Returns:
            [torch.Tensor]: The cropped batch
        """
        batch_size, n_channels, layer_width, layer_height = layer.size()
        xy1 = (layer_width - target_width) // 2
        xy2 = (layer_height - target_height) // 2
        return layer[:, :, xy1 : (xy1 + target_width), xy2 : (xy2 + target_height)]

    def forward(self, x: torch.Tensor, enable_concat: bool = True) -> torch.Tensor:
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
        crop = self.center_crop(contr_4, center.size()[2], center.size()[3])
        concat = torch.cat([center, crop * concat_weight], 1)

        expand = self.expand_4_2(self.expand_4_1(concat))
        upscale = self.upscale4(expand)

        crop = self.center_crop(contr_3, upscale.size()[2], upscale.size()[3])
        concat = torch.cat([upscale, crop * concat_weight], 1)

        expand = self.expand_3_2(self.expand_3_1(concat))
        upscale = self.upscale3(expand)

        crop = self.center_crop(contr_2, upscale.size()[2], upscale.size()[3])
        concat = torch.cat([upscale, crop * concat_weight], 1)

        expand = self.expand_2_2(self.expand_2_1(concat))
        upscale = self.upscale2(expand)

        crop = self.center_crop(contr_1, upscale.size()[2], upscale.size()[3])
        concat = torch.cat([upscale, crop * concat_weight], 1)

        expand = self.expand_1_2(self.expand_1_1(concat))

        if enable_concat:
            output = self.final(expand)
        else:
            output = self.output_reconstruction_map(expand)

        return output
