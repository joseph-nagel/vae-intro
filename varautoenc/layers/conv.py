'''Convolutional layers.'''

import torch.nn as nn

from .utils import make_conv, make_up


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
                 pool_last=True,
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
            conv = make_conv(
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

            # create upsampling layer
            up = make_up(
                scaling,
                mode=upsample_mode,
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size
            )

            layers.append(up)

            # create conv layer
            conv = make_conv(
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

