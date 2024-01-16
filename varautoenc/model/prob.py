'''Probabilistic decoder.'''

import torch.nn as nn

from ..layers import (
    DenseBlock,
    ConvUp,
    ProbConv
)


class ProbDecoder(nn.Module):
    '''Probabilistic decoder.'''

    def __init__(self,
                 num_features,
                 num_channels,
                 reshape,
                 kernel_size=3,
                 scaling=2,
                 upsample_mode='conv_transpose',
                 batchnorm=False,
                 activation='leaky_relu',
                 last_activation=None,
                 drop_rate=None,
                 sigma=None,
                 per_channel=False):

        super().__init__()

        self.reshape = reshape

        # create dense layers
        self.dense_layers = DenseBlock(
            num_features,
            activation=activation,
            last_activation='same',
            drop_rate=drop_rate
        )

        # create conv layers
        if len(num_channels) < 2:
            raise ValueError('Number of channels needs at least two entries')

        elif len(num_channels) == 2:
            self.conv_layers = None

        else:
            self.conv_layers = ConvUp(
                num_channels,
                kernel_size=kernel_size,
                padding='same',
                scaling=scaling,
                upsample_mode=upsample_mode,
                batchnorm=batchnorm,
                activation=activation,
                last_activation='same',
                normalize_last=True,
                conv_last=False # note that the last conv. is replaced by the mu/logsigma-layer below
            )

        # create mu and logsigma
        self.mu_logsigma = ProbConv(
            num_channels[-2],
            num_channels[-1],
            kernel_size=kernel_size,
            sigma=sigma,
            per_channel=per_channel,
            activation=last_activation
        )

    def forward(self, x):

        # run dense layers
        x = self.dense_layers(x)

        # reshape
        x = x.view(-1, *self.reshape)

        # run conv layers
        if self.conv_layers is not None:
            x = self.conv_layers(x)

        # predict mu and logsigma
        mu, logsigma = self.mu_logsigma(x)

        return mu, logsigma

