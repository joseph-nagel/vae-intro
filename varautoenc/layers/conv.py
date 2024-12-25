'''Convolutional layers.'''

from collections.abc import Sequence

import torch.nn as nn

from .utils import (
    IntOrInts,
    ActivType,
    make_activation,
    make_up
)


class SingleConv(nn.Sequential):
    '''Single conv. block.'''

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: IntOrInts = 3,
        stride: IntOrInts = 1,
        padding: IntOrInts | str = 'same',
        bias: bool = True,
        activation: ActivType | None = 'leaky_relu',
        batchnorm: bool = False
    ) -> None:

        # create conv layer
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias # the bias should be disabled if a batchnorm directly follows after the convolution
        )

        # create activation function
        activ = make_activation(activation)

        # create normalization
        norm = nn.BatchNorm2d(out_channels) if batchnorm else None

        # assemble block
        layers = [conv, activ, norm] # note that the normalization follows the activation (which could be reversed of course)
        not_none_layers = [l for l in layers if l is not None]

        # initialize module
        super().__init__(*not_none_layers)


class DoubleConv(nn.Sequential):
    '''Double conv. blocks.'''

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: IntOrInts = 3,
        stride: IntOrInts = 1,
        padding: IntOrInts | str = 'same',
        bias: bool = True,
        activation: ActivType | None = 'leaky_relu',
        last_activation: ActivType | None = 'same',
        batchnorm: bool = False,
        normalize_last: bool = True,
        inout_first: bool = True
    ) -> None:

        # determine last activation
        if last_activation == 'same':
            last_activation = activation

        # create first conv
        conv_block1 = SingleConv(
            in_channels,
            out_channels if inout_first else in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            activation=activation,
            batchnorm=batchnorm
        )

        # create second conv
        conv_block2 = SingleConv(
            out_channels if inout_first else in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=bias,
            activation=last_activation,
            batchnorm=(batchnorm and normalize_last)
        )

        # initialize module
        super().__init__(conv_block1, conv_block2)


class ConvBlock(nn.Sequential):
    '''Multiple (serial) conv. blocks.'''

    def __init__(
        self,
        num_channels: Sequence[int],
        kernel_size: IntOrInts = 3,
        stride: IntOrInts = 1,
        padding: IntOrInts | str = 'same',
        bias: bool = True,
        activation: ActivType | None = 'leaky_relu',
        last_activation: ActivType | None = 'same',
        batchnorm: bool = False,
        normalize_last: bool = True
    ) -> None:

        # determine last activation
        if last_activation == 'same':
            last_activation = activation

        # check number of layers
        if len(num_channels) >= 2:
            num_layers = len(num_channels) - 1
        else:
            raise ValueError('Number of channels needs at least two entries')

        # assemble layers
        layers = []

        for idx, (in_channels, out_channels) in enumerate(zip(num_channels[:-1], num_channels[1:])):
            is_not_last = (idx < num_layers - 1)

            # create conv layer
            conv_block = SingleConv(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
                activation=activation if is_not_last else last_activation,
                batchnorm=batchnorm if is_not_last else (batchnorm and normalize_last)
            )

            layers.append(conv_block)

        # initialize module
        super().__init__(*layers)


class ConvDown(nn.Sequential):
    '''Convolutions with downsampling.'''

    def __init__(
        self,
        num_channels: Sequence[int],
        kernel_size: IntOrInts = 3,
        padding: IntOrInts | str = 'same',
        stride: IntOrInts = 1,
        pooling: IntOrInts | None = 2,
        activation: ActivType | None = 'leaky_relu',
        last_activation: ActivType | None = 'same',
        batchnorm: bool = False,
        normalize_last: bool = True,
        pool_last: bool = True,
        double_conv: bool = False,
        inout_first: bool = True
    ) -> None:

        # determine conv type
        ConvType = DoubleConv if double_conv else SingleConv

        # create specific options
        kwargs = {'inout_first': inout_first} if double_conv else {}

        # determine last activation
        if last_activation == 'same':
            last_activation = activation

        # check number of layers
        if len(num_channels) >= 2:
            num_layers = len(num_channels) - 1
        else:
            raise ValueError('Number of channels needs at least two entries')

        # assemble layers
        layers = [] # type: list[nn.Module]

        for idx, (in_channels, out_channels) in enumerate(zip(num_channels[:-1], num_channels[1:])):
            is_not_last = (idx < num_layers - 1)

            # create conv layer
            conv = ConvType(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                activation=activation if is_not_last else last_activation,
                batchnorm=batchnorm if is_not_last else (batchnorm and normalize_last),
                **kwargs
            )

            layers.append(conv)

            # create pooling layer
            if pooling is not None:
                if is_not_last or pool_last:
                    down = nn.MaxPool2d(pooling)

                    layers.append(down)

        # initialize module
        super().__init__(*layers)


class ConvUp(nn.Sequential):
    '''Convolutions with upsampling.'''

    def __init__(
        self,
        num_channels: Sequence[int],
        kernel_size: IntOrInts = 3,
        padding: IntOrInts | str = 'same',
        scaling: int = 2,
        upsample_mode: str = 'conv_transpose',
        activation: ActivType | None = 'leaky_relu',
        last_activation: ActivType | None = 'same',
        batchnorm: bool = False,
        normalize_last: bool = True,
        conv_last: bool = True,
        up_first: bool = True,
        double_conv: bool = False,
        inout_first: bool = True
    ) -> None:

        # determine conv type
        ConvType = DoubleConv if double_conv else SingleConv

        # create specific options
        kwargs = {'inout_first': inout_first} if double_conv else {}

        # determine last activation
        if last_activation == 'same':
            last_activation = activation

        # check number of layers
        if len(num_channels) >= 2:
            num_layers = len(num_channels) - 1
        else:
            raise ValueError('Number of channels needs at least two entries')

        # assemble layers
        layers = [] # type: list[nn.Module]

        for idx, (in_channels, out_channels) in enumerate(zip(num_channels[:-1], num_channels[1:])):
            is_not_first = (idx > 0)
            is_not_last = (idx < num_layers - 1)

            # create upsampling layer
            if is_not_first or up_first:
                up = make_up(
                    scaling,
                    mode=upsample_mode,
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=kernel_size
                )

                layers.append(up)

            # create conv layer
            if is_not_last or conv_last:
                conv = ConvType(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    activation=activation if is_not_last else last_activation,
                    batchnorm=batchnorm if is_not_last else (batchnorm and normalize_last),
                    **kwargs
                )

                layers.append(conv)

        # initialize module
        super().__init__(*layers)

