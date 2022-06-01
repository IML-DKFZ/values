import torch
import torch.nn as nn


class DownConvolution(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: torch.nn.modules.activation,
        kernel_size_down: int = 2,
        kernel_size: int = 5,
        n_conv: int = 2,
    ):
        """
        Down convolution (compression path) which reduces the image size and increases the channels
        Args:
            in_channels (int): Number of input channels to this down convolution step
            out_channels (int): Number of output channels to the down convolution step
            activation (torch.nn.modules.activation): activation function that is used
            kernel_size_down (int): The kernel size for the down convolution
            kernel_size (int): The kernel size for the convolution blocks
            n_conv (int): The number of convolutions that is performed in this down convolution step
        """
        super(DownConvolution, self).__init__()
        self.n_conv = n_conv
        self.activation = activation
        self.down_conv = self.conv_block(
            in_channels, out_channels, kernel_size_down, 2, 0
        )
        self.conv_1 = self.conv_block(out_channels, out_channels, kernel_size, 1, 2)
        self.conv_2 = self.conv_block(out_channels, out_channels, kernel_size, 1, 2)

    def conv_block(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> nn.Sequential:
        """
        Convolution block consisting of 3D Convolution, BatchNorm and Activation function
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int): Size of the kernel used for convolution
            stride (int): The stride used in the convolution
            padding (int): Amount of padding
        Returns:
            layer (nn.Sequential): The convolution block to pass the input through
        """
        layer = nn.Sequential(
            nn.Conv3d(
                in_channels, out_channels, kernel_size, stride=stride, padding=padding
            ),
            nn.BatchNorm3d(out_channels),
            self.activation,
        )
        return layer

    def forward(self, x: torch.Tensor) -> torch:
        """
        Forward pass through a down convolution block
        Args:
            x (torch.Tensor): Input tensor
        Returns:
            output (torch.Tensor): Output of the down convolution block
        """
        down_conv = self.down_conv(x)
        conv = self.conv_1(down_conv)
        for conv_idx in range(1, self.n_conv):
            conv = self.conv_2(conv)
        residual = down_conv + conv
        output = self.activation(residual)
        return output


class UpConvolution(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int,
        activation,
        kernel_size_up: int = 2,
        kernel_size: int = 5,
        n_conv=2,
    ):
        """
        Up convolution (expansion path) which increases the image size and reduces the channels
        Args:
            in_channels (int): Number of input channels to this up convolution step
            out_channels (int): Number of output channels to the up convolution step
            skip_channels (int): Number of channels of the skip connection that is concatenated to the input
            activation (torch.nn.modules.activation): activation function that is used
            kernel_size_up (int): The kernel size for the up convolution
            kernel_size (int): The kernel size for the convolution blocks
            n_conv (int): The number of convolutions that is performed in this up convolution step
        """
        super(UpConvolution, self).__init__()
        self.n_conv = n_conv
        self.activation = activation
        self.up_conv = self.up_convolution(in_channels, skip_channels, kernel_size_up)
        self.conv_1 = self.conv_block(skip_channels * 2, out_channels, kernel_size, 1)
        self.conv_2 = self.conv_block(out_channels, out_channels, kernel_size, 1)

    def up_convolution(
        self, in_channels: int, out_channels: int, kernel_size: int
    ) -> nn.Sequential:
        """
        Up convolution consisting of transposed convolution, BatchNorm and activation
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int): Size of the kernel used for convolution
        Returns:
            nn.Sequential: The convolution block to pass the input through
        """
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=2),
            nn.BatchNorm3d(out_channels),
            self.activation,
        )

    def conv_block(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int
    ) -> nn.Sequential:
        """
        Convolution block consisting of 3D Convolution, BatchNorm and Activation function
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int): Size of the kernel used for convolution
            stride (int): The stride used in the convolution
        Returns:
            layer (nn.Sequential): The convolution block to pass the input through
        """
        layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=2),
            nn.BatchNorm3d(out_channels),
            self.activation,
        )
        return layer

    def forward(self, x, skip):
        """
        Forward pass through a up convolution block
        Args:
            x (torch.Tensor): Input tensor
            skip (torch.Tensor): Skip connection to concatenate
        Returns:
            output (torch.Tensor): Output of the up convolution block
        """
        up_conv = self.up_conv(x)
        concat = torch.cat((up_conv, skip), 1)
        conv = self.conv_1(concat)
        for conv_idx in range(1, self.n_conv):
            conv = self.conv_2(conv)
        residual = concat + conv
        output = self.activation(residual)
        return output


class VNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 1,
        initial_filter_size: int = 16,
        kernel_size: int = 5,
        kernel_size_down=2,
        kernel_size_up: int = 2,
        activation_function="elu",
    ):
        """VNet (torch.nn.Module) for 3D input segmentation

        Args:
            num_classes ([int]): [Number of output classes for segmentation]
            in_channels (int, optional): [num in_channels]. Defaults to 1.
            initial_filter_size (int, optional): [number of filters for factor]. Defaults to 16.
            kernel_size (int, optional): [size of kernels]. Defaults to 5.
            kernel_size_down (int, optional): [size of kernels for down convolution]. Defaults to 2.
            kernel_size_up (int, optional): [size of kernels for up convolution]. Defaults to 2.
            activation_function (int, optional): [activation function that is used]. Either "elu" or "prelu".
                                                 Defaults to "elu".
        """
        super(VNet, self).__init__()
        self.in_channels = in_channels
        self.in_conv = self.input_conv(in_channels, initial_filter_size, kernel_size)
        self.activation = nn.ELU() if activation_function == "elu" else nn.PReLU()
        # 16 to 32 channels
        self.down_conv_1 = DownConvolution(
            initial_filter_size,
            initial_filter_size * 2,
            kernel_size=kernel_size,
            kernel_size_down=kernel_size_down,
            activation=self.activation,
            n_conv=2,
        )
        # 32 to 64 channels
        self.down_conv_2 = DownConvolution(
            initial_filter_size * 2,
            initial_filter_size * 2**2,
            kernel_size=kernel_size,
            kernel_size_down=kernel_size_down,
            activation=self.activation,
            n_conv=3,
        )
        # 64 to 128 channels
        self.down_conv_3 = DownConvolution(
            initial_filter_size * 2**2,
            initial_filter_size * 2**3,
            kernel_size=kernel_size,
            kernel_size_down=kernel_size_down,
            activation=self.activation,
            n_conv=3,
        )
        # 128 to 256 channels
        self.bottleneck = DownConvolution(
            initial_filter_size * 2**3,
            initial_filter_size * 2**4,
            kernel_size=kernel_size,
            kernel_size_down=kernel_size_down,
            activation=self.activation,
            n_conv=3,
        )

        # keep 256 channels
        self.up_conv_1 = UpConvolution(
            initial_filter_size * 2**4,
            initial_filter_size * 2**4,
            initial_filter_size * 2**3,
            kernel_size=kernel_size,
            kernel_size_up=kernel_size_up,
            activation=self.activation,
            n_conv=3,
        )

        # 256 to 128 channels
        self.up_conv_2 = UpConvolution(
            initial_filter_size * 2**4,
            initial_filter_size * 2**3,
            initial_filter_size * 2**2,
            kernel_size=kernel_size,
            kernel_size_up=kernel_size_up,
            activation=self.activation,
            n_conv=3,
        )

        # 128 to 64 channels
        self.up_conv_3 = UpConvolution(
            initial_filter_size * 2**3,
            initial_filter_size * 2**2,
            initial_filter_size * 2,
            kernel_size=kernel_size,
            kernel_size_up=kernel_size_up,
            activation=self.activation,
            n_conv=2,
        )

        # 64 to 32 channels
        self.up_conv_4 = UpConvolution(
            initial_filter_size * 2**2,
            initial_filter_size * 2,
            initial_filter_size,
            kernel_size=kernel_size,
            kernel_size_up=kernel_size_up,
            activation=self.activation,
            n_conv=1,
        )

        self.out_conv = self.output_conv(initial_filter_size * 2, num_classes)

        self.softmax = nn.Softmax()
        return

    @staticmethod
    def input_conv(
        in_channels: int, out_channels: int, kernel_size: int = 5
    ) -> nn.Sequential:
        """
        Input convolution that is performed in the beginning of the forward pass consisting of convolution and BatchNorm
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int): Size of the kernel used for convolution
        Returns:
            layer (nn.Sequential): The convolution block to pass the input through
        """
        layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=2),
            nn.BatchNorm3d(out_channels),
        )
        return layer

    def output_conv(
        self, in_channels: int, out_channels: int, kernel_size: int = 1
    ) -> nn.Sequential:
        """
        Output convolution that is performed at the end of the forward pass consisting of convolution, BatchNorm and
        activation
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int): Size of the kernel used for convolution
        Returns:
            layer (nn.Sequential): The convolution block to pass the input through
        """
        layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=1),
            nn.BatchNorm3d(out_channels),
            self.activation,
        )
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network

        Args:
            x (torch.Tensor): input batch

        Returns:
            out_conv (torch.Tensor): The result of the network
        """
        # Encoder
        in_conv = self.in_conv(x)
        repeat_num = 16 // self.in_channels
        x16 = x.repeat([1, repeat_num, 1, 1, 1][:5])
        residual_1 = self.activation(x16 + in_conv)
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
