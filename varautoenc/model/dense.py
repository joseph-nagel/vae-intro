'''Dense encoder/decoder.'''

import torch
import torch.nn as nn

from ..layers import (
    make_dense,
    DenseBlock,
    MultiDense,
    ProbDense
)


class DenseEncoder(nn.Module):
    '''Fully connected encoder.'''

    def __init__(self,
                 num_features,
                 batchnorm=False,
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
                batchnorm=batchnorm,
                activation=activation,
                last_activation='same',
                drop_rate=drop_rate
            )

        # create Gaussian param layers
        self.gaussian_params = MultiDense(
            num_features[-2],
            num_features[-1],
            num_outputs=2,
            batchnorm=False,
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

        # predict Gaussian params
        mu, logsigma = self.gaussian_params(x)

        return mu, logsigma


class DenseDecoder(nn.Module):
    '''Fully connected decoder.'''

    def __init__(self,
                 num_features,
                 batchnorm=False,
                 activation='leaky_relu',
                 drop_rate=None,
                 reshape=None,
                 likelihood_type='Bernoulli',
                 sigma=None,
                 per_feature=False):

        super().__init__()

        self.reshape = reshape

        # set likelihood type
        if likelihood_type in ('Bernoulli', 'Gauss', 'Gaussian'):
            self.likelihood_type = likelihood_type
        else:
            raise ValueError(f'Unknown likelihood type: {likelihood_type}')

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
                last_activation=None,
                drop_rate=drop_rate
            )

        # create Bernoulli logits
        if self.likelihood_type == 'Bernoulli':
            self.bernoulli_logits = make_dense(
                num_features[-2],
                num_features[-1],
                batchnorm=False,
                activation=None,
                drop_rate=drop_rate
            )

        # create Gaussian params
        else:
            self.gaussian_params = ProbDense(
                num_features[-2],
                num_features[-1],
                sigma=sigma,
                per_feature=per_feature,
                activation=None,
                drop_rate=drop_rate
            )

    def forward(self, x):

        # run dense layers
        if self.dense_layers is not None:
            x = self.dense_layers(x)

        # predict Bernoulli logits
        if self.likelihood_type == 'Bernoulli':
            logits = self.bernoulli_logits(x)

            # reshape
            if self.reshape is not None:
                logits = logits.view(-1, *self.reshape)

            return logits

        # predict Gaussian params
        else:
            mu, logsigma = self.gaussian_params(x)

            # reshape
            if self.reshape is not None:
                mu = mu.view(-1, *self.reshape)

                if logsigma.numel() == 1:
                    logsigma = logsigma.view(*[1 for _ in range(len(self.reshape))]) # expand single sigma
                else:
                    logsigma = logsigma.view(*self.reshape) # reshape feature-specific logsigmas

            return mu, logsigma

