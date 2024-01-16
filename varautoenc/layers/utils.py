'''Model layer utils.'''

from inspect import isfunction, isclass

import torch.nn as nn


ACTIVATIONS = {
    'none': None,
    'identity': nn.Identity,
    'sigmoid': nn.Sigmoid,
    'tanh': nn.Tanh,
    'relu': nn.ReLU,
    'leaky_relu': nn.LeakyReLU,
    'elu': nn.ELU,
    'softplus': nn.Softplus,
    'swish': nn.SiLU
}


def make_activation(mode='leaky_relu'):
    '''Create activation function.'''

    if mode is None:
        activ = None

    elif mode in ACTIVATIONS.keys():

        a = ACTIVATIONS[mode]

        if isfunction(a):
            activ = a
        elif isclass(a):
            activ = a()
        else:
            activ = a

    else:
        raise ValueError(f'Unknown activation: {mode}')

    return activ


def make_block(layers):
    '''Assemble a block of layers.'''

    if isinstance(layers, nn.Module):
        block = layers

    elif isinstance(layers, (list, tuple)):

        layers = [l for l in layers if l is not None]

        if len(layers) == 1:
            block = layers[0]
        else:
            block = nn.Sequential(*layers)

    else:
        raise TypeError(f'Invalid layers type: {type(layers)}')

    return block


def make_dropout(drop_rate=None):
    '''Create a dropout layer.'''

    if drop_rate is None:
        dropout = None
    else:
        dropout = nn.Dropout(p=drop_rate)

    return dropout


def make_dense(in_features,
               out_features,
               bias=True,
               activation=None,
               drop_rate=None):
    '''
    Create fully connected layer.

    Parameters
    ----------
    in_features : int
        Number of inputs.
    out_features : int
        Number of outputs.
    bias : bool
        Determines whether a bias is used.
    activation : None or str
        Determines the nonlinearity.
    drop_rate : float
        Dropout probability.

    '''

    # create dropout layer
    dropout = make_dropout(drop_rate=drop_rate)

    # create dense layer
    linear = nn.Linear(in_features, out_features, bias=bias)

    # create activation function
    activation = make_activation(activation)

    # assemble block
    layers = [dropout, linear, activation]
    dense_block = make_block(layers)

    return dense_block


def make_conv(in_channels,
              out_channels,
              kernel_size=3,
              stride=1,
              padding='same',
              bias=True,
              batchnorm=False,
              activation=None):
    '''
    Create convolutional layer.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Convolutional kernel size.
    stride : int
        Stride parameter.
    padding : int
        Padding parameter.
    bias : bool
        Determines whether a bias is used.
    batchnorm : None or str
        Determines whether batchnorm is used.
    activation : None or str
        Determines the nonlinearity.

    '''

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
    conv_block = make_block(layers)

    return conv_block


def make_up(scaling,
            mode='conv_transpose',
            in_channels=1,
            out_channels=1,
            kernel_size=3):
    '''Create upsampling layer.'''

    # bilinear upsampling
    if mode == 'bilinear':
        up = nn.Upsample(
            scale_factor=scaling,
            mode='bilinear',
            align_corners=True
        )

    # bilinear upsampling followed by a convolution
    elif mode == 'bilinear_conv':
        up = nn.Sequential(
            nn.Upsample(
                scale_factor=scaling,
                mode='bilinear',
                align_corners=True
            ),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding='same'
            )
        )

    # transposed convolution
    elif mode == 'conv_transpose':
        up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=scaling,
            stride=scaling,
            padding=0
        )

    else:
        raise ValueError(f'Unknown upsample mode: {mode}')

    return up

