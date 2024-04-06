'''Convolutional encoder/decoder.'''

import torch
import torch.nn as nn

from ..layers import (
    DenseBlock,
    MultiDense,
    SingleConv,
    DoubleConv,
    ConvDown,
    ConvUp,
    ProbConv
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
                 drop_rate=None,
                 pool_last=True,
                 double_conv=True,
                 inout_first=True):

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
            normalize_last=True,
            pool_last=pool_last,
            double_conv=double_conv,
            inout_first=inout_first
        )

        # create dense layers
        if len(num_features) < 2:
            raise ValueError('Number of features needs at least two entries')

        elif len(num_features) == 2:
            self.dense_layers = None

        else:
            self.dense_layers = DenseBlock(
                num_features[:-1], # the last layer is replaced by the prob. layer below
                batchnorm=batchnorm,
                activation=activation,
                last_activation='same',
                normalize_last=True,
                drop_rate=drop_rate
            )

        # create Gaussian param layers
        self.dist_params = MultiDense(
            num_features[-2],
            num_features[-1],
            num_outputs=2,
            batchnorm=False,
            activation=None,
            drop_rate=drop_rate
        )

    def forward(self, x):

        # run conv layers
        x = self.conv_layers(x)

        # flatten
        x = torch.flatten(x, start_dim=1)

        # run dense layers
        if self.dense_layers is not None:
            x = self.dense_layers(x)

        # predict Gaussian params
        mu, logsigma = self.dist_params(x)

        return mu, logsigma


class ConvDecoder(nn.Module):
    '''Convolutional decoder.'''

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
                 up_first=True,
                 double_conv=True,
                 inout_first=True,
                 likelihood_type='Bernoulli',
                 sigma=None,
                 per_channel=False):

        super().__init__()

        self.reshape = reshape

        self.likelihood_type = likelihood_type

        # create dense layers
        self.dense_layers = DenseBlock(
            num_features,
            batchnorm=batchnorm,
            activation=activation,
            last_activation='same',
            normalize_last=True,
            drop_rate=drop_rate
        )

        # create conv layers
        if len(num_channels) < 2:
            raise ValueError('Number of channels needs at least two entries')

        else:
            self.conv_layers = ConvUp(
                num_channels, # the last channel is passed for upscaling purposes
                kernel_size=kernel_size,
                padding='same',
                scaling=scaling,
                upsample_mode=upsample_mode,
                batchnorm=batchnorm,
                activation=activation,
                last_activation='same',
                normalize_last=True,
                conv_last=False, # the last layer is replaced by the prob. layer below
                up_first=up_first,
                double_conv=double_conv,
                inout_first=inout_first
            )

        # create last layer options
        kwargs = {
            'kernel_size': kernel_size,
            'stride': 1,
            'padding': 'same'
        }

        if double_conv:
            kwargs = {
                **kwargs,
                'batchnorm': batchnorm,
                'activation': activation,
                'last_activation': last_activation,
                'normalize_last': False,
                'inout_first': inout_first
            }
        else:
            kwargs = {
                **kwargs,
                'batchnorm': False,
                'activation': last_activation,
            }

        # create Bernoulli logits
        if self.likelihood_type in ('Bernoulli', 'ContinuousBernoulli'):
            ConvType = DoubleConv if double_conv else SingleConv

            self.bernoulli_logits = ConvType(
                num_channels[-2],
                num_channels[-1],
                **kwargs
            )

        # create Gaussian/Laplace params
        else:
            self.dist_params = ProbConv(
                num_channels[-2],
                num_channels[-1],
                double_conv=double_conv,
                sigma=sigma,
                per_channel=per_channel,
                **kwargs
            )

    def forward(self, x):

        # run dense layers
        x = self.dense_layers(x)

        # reshape
        x = x.view(-1, *self.reshape)

        # run conv layers
        x = self.conv_layers(x)

        # predict Bernoulli logits
        if self.likelihood_type in ('Bernoulli', 'ContinuousBernoulli'):
            logits = self.bernoulli_logits(x)
            return logits

        # predict Gaussian/Laplace params
        else:
            mu, logsigma = self.dist_params(x)
            return mu, logsigma

