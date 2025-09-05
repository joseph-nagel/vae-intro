'''Convolutional encoder/decoder.'''

from collections.abc import Sequence

import torch
import torch.nn as nn

from ..layers import (
    IntOrInts,
    ActivType,
    SigmaType,
    DenseBlock,
    MultiDense,
    SingleConv,
    DoubleConv,
    ConvDown,
    ConvUp,
    ProbConv
)


class ConvEncoder(nn.Module):
    '''
    Convolutional encoder.

    Parameters
    ----------
    num_channels : list or tuple
        Channel numbers for conv. layers.
    num_features : list or tuple
        Feature numbers for dense layers.
    kernel_size : int or (int, int)
        Conv. kernel size.
    pooling : int, (int, int) or None
        Pooling parameter.
    activation : str or None
        Nonlinearity type.
    batchnorm : bool
        Determines whether batchnorm is used.
    drop_rate : float or None
        Dropout probability for dense layers.
    pool_last : bool
        Controls the last pooling operation (also first upscaling).
    double_conv : bool
        Determines whether double conv. blocks are used.
    inout_first : bool
        Determines whether the first or second
        convolution changes the number of channels.

    '''

    def __init__(
        self,
        num_channels: Sequence[int],
        num_features: Sequence[int],
        kernel_size: IntOrInts = 3,
        pooling: IntOrInts | None = 2,
        activation: ActivType | None = 'leaky_relu',
        batchnorm: bool = True,
        drop_rate: float | None = None,
        pool_last: bool = True,
        double_conv: bool = True,
        inout_first: bool = True
    ) -> None:

        super().__init__()

        # create conv layers
        self.conv_layers = ConvDown(
            num_channels,
            kernel_size=kernel_size,
            stride=1,
            padding='same',
            pooling=pooling,
            activation=activation,
            last_activation='same',
            batchnorm=batchnorm,
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
                num_features[:-1],  # the last layer is replaced by the prob. layer below
                activation=activation,
                last_activation='same',
                batchnorm=batchnorm,
                normalize_last=True,
                drop_rate=drop_rate
            )

        # create Gaussian param layers
        self.dist_params = MultiDense(
            num_features[-2],
            num_features[-1],
            num_blocks=2,
            activation=None,
            batchnorm=False,
            drop_rate=drop_rate
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

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
    '''
    Convolutional decoder.

    Parameters
    ----------
    num_features : list or tuple
        Feature numbers for dense layers.
    num_channels : list or tuple
        Channel numbers for conv. layers.
    reshape : list or tuple
        Shape between dense and conv. layers.
    kernel_size : int or (int, int)
        Conv. kernel size.
    scaling : int
        Scaling parameter.
    upsample_mode : {'bilinear', 'bilinear_conv', 'conv_transpose'}
        Conv. upsampling mode.
    activation : str or None
        Nonlinearity type.
    last_activation : str or None
        Nonlinearity of the final layer.
    batchnorm : bool
        Determines whether batchnorm is used.
    drop_rate : float or None
        Dropout probability for dense layers.
    up_first : bool
        Determines whether upsampling is done first.
    double_conv : bool
        Determines whether double conv. blocks are used.
    inout_first : bool
        Determines whether the first or second
        convolution changes the number of channels.
    likelihood_type : {'Bernoulli', 'ContinuousBernoulli', 'Gauss', 'Gaussian', 'Laplace'}
        Likelihood function type.
    sigma : float or None
        Can be used to specify constant sigmas.
    per_channel : bool
        Enables channel-specific sigma parameters.

    '''

    def __init__(
        self,
        num_features: Sequence[int],
        num_channels: Sequence[int],
        reshape: Sequence[int],
        kernel_size: IntOrInts = 3,
        scaling: int = 2,
        upsample_mode: str = 'conv_transpose',
        activation: ActivType | None = 'leaky_relu',
        last_activation: ActivType | None = None,
        batchnorm: bool = False,
        drop_rate: float | None = None,
        up_first: bool = True,
        double_conv: bool = True,
        inout_first: bool = True,
        likelihood_type: str = 'Bernoulli',
        sigma: SigmaType | None = None,
        per_channel: bool = False
    ) -> None:

        super().__init__()

        self.reshape = reshape

        self.likelihood_type = likelihood_type

        # create dense layers
        self.dense_layers = DenseBlock(
            num_features,
            activation=activation,
            last_activation='same',
            batchnorm=batchnorm,
            normalize_last=True,
            drop_rate=drop_rate
        )

        # create conv layers
        if len(num_channels) < 2:
            raise ValueError('Number of channels needs at least two entries')

        else:
            self.conv_layers = ConvUp(
                num_channels,  # the last channel is passed for upscaling purposes
                kernel_size=kernel_size,
                padding='same',
                scaling=scaling,
                upsample_mode=upsample_mode,
                activation=activation,
                last_activation='same',
                batchnorm=batchnorm,
                normalize_last=True,
                conv_last=False,  # the last layer is replaced by the prob. layer below
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
                'activation': activation,
                'last_activation': last_activation,
                'batchnorm': batchnorm,
                'normalize_last': False,
                'inout_first': inout_first
            }
        else:
            kwargs = {
                **kwargs,
                'activation': last_activation,
                'batchnorm': False
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

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:

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
