'''Convolutional layers.'''

import torch.nn as nn

from .utils import make_activation, make_up


class SingleConv(nn.Sequential):
    '''Single conv. block.'''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding='same',
                 bias=True,
                 batchnorm=False,
                 activation='leaky_relu'):

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
        activation = make_activation(activation)

        # create normalization
        norm = nn.BatchNorm2d(out_channels) if batchnorm else None

        # assemble block
        layers = [conv, activation, norm] # note that the normalization follows the activation (which could be reversed of course)
        layers = [l for l in layers if l is not None]

        # initialize module
        super().__init__(*layers)


class DoubleConv(nn.Sequential):
    '''Double conv. blocks.'''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding='same',
                 bias=True,
                 batchnorm=False,
                 activation='leaky_relu',
                 last_activation='same',
                 normalize_last=True):

        # determine last activation
        if last_activation == 'same':
            last_activation = activation

        # create first conv
        conv_block1 = SingleConv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            batchnorm=batchnorm,
            activation=activation
        )

        # create second conv
        conv_block2 = SingleConv(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=bias,
            batchnorm=(batchnorm and normalize_last),
            activation=last_activation
        )

        # initialize module
        super().__init__(conv_block1, conv_block2)


class ConvBlock(nn.Sequential):
    '''Multiple (serial) conv. blocks.'''

    def __init__(self,
                 num_channels,
                 kernel_size=3,
                 stride=1,
                 padding='same',
                 bias=True,
                 batchnorm=False,
                 activation='leaky_relu',
                 last_activation='same',
                 normalize_last=True):

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
                batchnorm=batchnorm if is_not_last else (batchnorm and normalize_last),
                activation=activation if is_not_last else last_activation
            )

            layers.append(conv_block)

        # initialize module
        super().__init__(*layers)


class ConvDown(nn.Sequential):
    '''Convolutions with downsampling.'''

    def __init__(self,
                 num_channels,
                 kernel_size=3,
                 padding='same',
                 stride=1,
                 pooling=2,
                 batchnorm=False,
                 activation='leaky_relu',
                 last_activation='same',
                 normalize_last=True,
                 pool_last=True,
                 double_conv=False):

        # determine conv type
        ConvType = DoubleConv if double_conv else SingleConv

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
            conv = ConvType(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                batchnorm=batchnorm if is_not_last else (batchnorm and normalize_last),
                activation=activation if is_not_last else last_activation
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

    def __init__(self,
                 num_channels,
                 kernel_size=3,
                 padding='same',
                 scaling=2,
                 upsample_mode='conv_transpose',
                 batchnorm=False,
                 activation='leaky_relu',
                 last_activation='same',
                 normalize_last=True,
                 conv_last=True,
                 up_first=True,
                 double_conv=False):

        # determine conv type
        ConvType = DoubleConv if double_conv else SingleConv

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
                    batchnorm=batchnorm if is_not_last else (batchnorm and normalize_last),
                    activation=activation if is_not_last else last_activation
                )

                layers.append(conv)

        # initialize module
        super().__init__(*layers)

