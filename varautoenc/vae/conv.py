'''Convolutional encoder/decoder.'''

import torch
import torch.nn as nn

from ..layers import (
    DenseModel,
    MultiDense,
    ConvDown,
    ConvUp
)

class ConvEncoder(nn.Module):
    '''Convolutional encoder.'''

    def __init__(self,
                 num_channels,
                 num_features,
                 kernel_size=3,
                 pooling=2,
                 batchnorm=True,
                 activation='leaky_relu',
                 pool_last=True):

        super().__init__()

        # create conv layers
        self.conv_layers = ConvDown(
            num_channels,
            kernel_size=kernel_size,
            padding='same',
            stride=1,
            pooling=pooling,
            batchnorm=batchnorm,
            activation=activation,
            last_activation='same',
            pool_last=pool_last,
            normalize_last=True
        )

        # create dense layers
        if len(num_features) < 2:
            raise ValueError('Number of features needs at least two entries')

        elif len(num_features) == 2:
            self.dense_layers = None

        else:
            self.dense_layers = DenseModel(
                num_features=num_features[:-1],
                activation=activation,
                last_activation='same'
            )

        # create mu and logsigma
        self.mu_logsigma = MultiDense(
            num_features[-2],
            num_features[-1],
            num_outputs=2
        )

    def forward(self, x):

        # run conv layers
        x = self.conv_layers(x)

        # flatten
        x = torch.flatten(x, start_dim=1)

        # run dense layers
        if self.dense_layers is not None:
            x = self.dense_layers(x)

        # predict mu and logsigma
        mu, logsigma = self.mu_logsigma(x)

        return mu, logsigma


class ConvDecoder(nn.Module):
    '''Convolutional decoder.'''

    def __init__(self,
                 num_features,
                 num_channels,
                 target_shape,
                 kernel_size=3,
                 scaling=2,
                 upsample_mode='conv_transpose',
                 batchnorm=False,
                 activation='leaky_relu',
                 last_activation=None):

        super().__init__()

        self.target_shape = target_shape

        # create dense layers
        self.dense_layers = DenseModel(
            num_features=num_features,
            activation=activation,
            last_activation='same'
        )

        # create conv layers
        self.conv_layers = ConvUp(
            num_channels,
            kernel_size=kernel_size,
            padding='same',
            scaling=scaling,
            upsample_mode=upsample_mode,
            batchnorm=batchnorm,
            activation=activation,
            last_activation=last_activation,
            normalize_last=False
        )

    def forward(self, x):

        # run dense layers
        x = self.dense_layers(x)

        # reshape
        x = x.view(-1, *self.target_shape)

        # run conv layers
        x = self.conv_layers(x)

        return x

