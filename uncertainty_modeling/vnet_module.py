import torch
import torch.nn as nn


class DownConvolution(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels,
        kernel_size_down: int = 2,
        kernel_size: int = 5,
        n_conv: int = 2,
    ):
        super(DownConvolution, self).__init__()
        self.n_conv = n_conv
        self.down_conv = self.conv_block(
            in_channels, out_channels, kernel_size_down, 2, 0
        )
        self.conv_1 = self.conv_block(out_channels, out_channels, kernel_size, 1, 2)
        self.conv_2 = self.conv_block(out_channels, out_channels, kernel_size, 1, 2)

    @staticmethod
    def conv_block(
        in_channels, out_channels, kernel_size, stride, padding
    ) -> nn.Sequential:
        layer = nn.Sequential(
            nn.Conv3d(
                in_channels, out_channels, kernel_size, stride=stride, padding=padding
            ),
            nn.BatchNorm3d(out_channels),
            nn.PReLU(),
        )
        return layer

    def forward(self, x):
        down_conv = self.down_conv(x)
        conv = self.conv_1(down_conv)
        for conv_idx in range(1, self.n_conv):
            conv = self.conv_2(conv)
        residual = down_conv + conv
        return residual


class UpConvolution(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int,
        kernel_size_up: int = 2,
        kernel_size: int = 5,
        n_conv=2,
    ):
        super(UpConvolution, self).__init__()
        self.n_conv = n_conv
        self.up_conv = self.up_convolution(in_channels, out_channels, kernel_size_up)
        self.conv_1 = self.conv_block(
            out_channels + skip_channels, out_channels, kernel_size, 1
        )
        self.conv_2 = self.conv_block(out_channels, out_channels, kernel_size, 1)

    @staticmethod
    def up_convolution(in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=2),
            nn.BatchNorm3d(out_channels),
            nn.PReLU(),
        )

    @staticmethod
    def conv_block(in_channels, out_channels, kernel_size, stride) -> nn.Sequential:
        layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=2),
            nn.BatchNorm3d(out_channels),
            nn.PReLU(),
        )
        return layer

    def forward(self, x, skip):
        up_conv = self.up_conv(x)
        concat = torch.cat((up_conv, skip), 1)
        conv = self.conv_1(concat)
        for conv_idx in range(1, self.n_conv):
            conv = self.conv_2(conv)
        residual = up_conv + conv
        return residual


class VNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 1,
        initial_filter_size: int = 16,
        kernel_size: int = 5,
        kernel_size_down=2,
    ):
        super(VNet, self).__init__()
        self.in_conv = self.input_conv(in_channels, initial_filter_size, kernel_size)
        # 16 to 32 channels
        self.down_conv_1 = DownConvolution(
            initial_filter_size, initial_filter_size * 2, n_conv=2
        )
        # 32 to 64 channels
        self.down_conv_2 = DownConvolution(
            initial_filter_size * 2, initial_filter_size * 2**2, n_conv=3
        )
        # 64 to 128 channels
        self.down_conv_3 = DownConvolution(
            initial_filter_size * 2**2, initial_filter_size * 2**3, n_conv=3
        )
        # 128 to 256 channels
        self.bottleneck = DownConvolution(
            initial_filter_size * 2**3, initial_filter_size * 2**4, n_conv=3
        )

        # keep 256 channels
        self.up_conv_1 = UpConvolution(
            initial_filter_size * 2**4,
            initial_filter_size * 2**4,
            initial_filter_size * 2**3,
            n_conv=3,
        )

        # 256 to 128 channels
        self.up_conv_2 = UpConvolution(
            initial_filter_size * 2**4,
            initial_filter_size * 2**3,
            initial_filter_size * 2**2,
            n_conv=3,
        )

        # 128 to 64 channels
        self.up_conv_3 = UpConvolution(
            initial_filter_size * 2**3,
            initial_filter_size * 2**2,
            initial_filter_size * 2,
            n_conv=2,
        )

        # 64 to 32 channels
        self.up_conv_4 = UpConvolution(
            initial_filter_size * 2**2,
            initial_filter_size * 2,
            initial_filter_size,
            n_conv=1,
        )

        self.out_conv = self.output_conv(initial_filter_size * 2, num_classes)

        self.softmax = nn.Softmax()
        return

    @staticmethod
    def input_conv(
        in_channels: int, out_channels: int, kernel_size: int = 5
    ) -> nn.Sequential:
        layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=2),
            nn.BatchNorm3d(out_channels),
            nn.PReLU(),
        )
        return layer

    @staticmethod
    def output_conv(
        in_channels: int, out_channels: int, kernel_size: int = 1
    ) -> nn.Sequential:
        layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=1),
            nn.BatchNorm3d(out_channels),
            nn.PReLU(),
        )
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        in_conv = self.in_conv(x)
        residual_1 = x + in_conv
        down_conv_1 = self.down_conv_1(residual_1)
        down_conv_2 = self.down_conv_2(down_conv_1)
        down_conv_3 = self.down_conv_3(down_conv_2)
        # Bottleneck
        bottleneck = self.bottleneck(down_conv_3)
        # Decoder
        up_conv_1 = self.up_conv_1(bottleneck, down_conv_3)
        up_conv_2 = self.up_conv_2(up_conv_1, down_conv_2)
        up_conv_3 = self.up_conv_3(up_conv_2, down_conv_1)
        up_conv_4 = self.up_conv_4(up_conv_3, residual_1)
        out_conv = self.out_conv(up_conv_4)
        return out_conv
