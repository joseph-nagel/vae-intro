'''Dense encoder/decoder.'''

import torch
import torch.nn as nn

from ..layers import DenseBlock, MultiDense


class DenseEncoder(nn.Module):
    '''Fully connected encoder.'''

    def __init__(self,
                 num_features,
                 activation='leaky_relu',
                 drop_rate=None,
                 flatten=True):

        super().__init__()

        self.flatten = flatten

        # create dense layers
        if len(num_features) < 2:
            raise ValueError('Number of features needs at least two entries')

        elif len(num_features) == 2:
            self.dense_layers = None

        else:
            self.dense_layers = DenseBlock(
                num_features[:-1],
                activation=activation,
                last_activation='same',
                drop_rate=drop_rate
            )

        # create mu and logsigma
        self.mu_logsigma = MultiDense(
            num_features[-2],
            num_features[-1],
            num_outputs=2,
            activation=None,
            drop_rate=drop_rate
        )

    def forward(self, x):

        # flatten
        if self.flatten:
            x = torch.flatten(x, start_dim=1)

        # run dense layers
        if self.dense_layers is not None:
            x = self.dense_layers(x)

        # predict mu and logsigma
        mu, logsigma = self.mu_logsigma(x)

        return mu, logsigma


class DenseDecoder(nn.Module):
    '''Fully connected decoder.'''

    def __init__(self,
                 num_features,
                 activation='leaky_relu',
                 drop_rate=None,
                 reshape=None):

        super().__init__()

        self.reshape = reshape

        self.dense_layers = DenseBlock(
            num_features,
            activation=activation,
            last_activation=None,
            drop_rate=drop_rate
        )

    def forward(self, x):

        # run dense layers
        x = self.dense_layers(x)

        # reshape
        if self.reshape is not None:
            x = x.view(-1, *self.reshape)

        return x

