'''Dense encoder/decoder.'''

from collections.abc import Sequence

import torch
import torch.nn as nn

from ..layers import (
    ActivType,
    SigmaType,
    make_dense,
    DenseBlock,
    MultiDense,
    ProbDense
)


class DenseEncoder(nn.Module):
    '''
    Fully connected encoder.

    Parameters
    ----------
    num_features : list of tuple
        Number of features.
    activation : str or None
        Nonlinearity type.
    batchnorm : bool
        Determines whether batchnorm is used.
    drop_rate : float or None
        Dropout probability.
    flatten : bool
        Determines whether input tensors are flattened
        before being further processed by the encoder.

    '''

    def __init__(
        self,
        num_features: Sequence[int],
        activation: ActivType | None = 'leaky_relu',
        batchnorm: bool = False,
        drop_rate: float | None = None,
        flatten: bool = True
    ) -> None:

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

        # flatten
        if self.flatten:
            x = torch.flatten(x, start_dim=1)

        # run dense layers
        if self.dense_layers is not None:
            x = self.dense_layers(x)

        # predict Gaussian params
        mu, logsigma = self.dist_params(x)

        return mu, logsigma


class DenseDecoder(nn.Module):
    '''
    Fully connected decoder.

    Parameters
    ----------
    num_features : list of tuple
        Number of features.
    activation : str or None
        Nonlinearity type.
    batchnorm : bool
        Determines whether batchnorm is used.
    drop_rate : float or None
        Dropout probability.
    reshape : list or None
        Final output shape.
    likelihood_type : {'Bernoulli', 'ContinuousBernoulli', 'Gauss', 'Gaussian', 'Laplace'}
        Likelihood function type.
    sigma : float or None
        Can be used to specify constant sigmas.
    per_feature : bool
        Enables feature-specific sigma parameters.

    '''

    def __init__(
        self,
        num_features: Sequence[int],
        activation: ActivType | None = 'leaky_relu',
        batchnorm: bool = False,
        drop_rate: float | None = None,
        reshape: Sequence[int] | None = None,
        likelihood_type: str = 'Bernoulli',
        sigma: SigmaType | None = None,
        per_feature: bool = False
    ) -> None:

        super().__init__()

        self.reshape = reshape

        self.likelihood_type = likelihood_type

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

        # create Bernoulli logits
        if self.likelihood_type in ('Bernoulli', 'ContinuousBernoulli'):
            self.bernoulli_logits = make_dense(
                num_features[-2],
                num_features[-1],
                activation=None,
                batchnorm=False,
                drop_rate=drop_rate
            )

        # create Gaussian/Laplace params
        else:
            self.dist_params = ProbDense(
                num_features[-2],
                num_features[-1],
                sigma=sigma,
                per_feature=per_feature,
                activation=None,
                batchnorm=False,
                drop_rate=drop_rate
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:

        # run dense layers
        if self.dense_layers is not None:
            x = self.dense_layers(x)

        # predict Bernoulli logits
        if self.likelihood_type in ('Bernoulli', 'ContinuousBernoulli'):
            logits = self.bernoulli_logits(x)

            # reshape
            if self.reshape is not None:
                logits = logits.view(-1, *self.reshape)

            return logits

        # predict Gaussian/Laplace params
        else:
            mu, logsigma = self.dist_params(x)

            # reshape
            if self.reshape is not None:
                mu = mu.view(-1, *self.reshape)

                if logsigma.numel() == 1:
                    logsigma = logsigma.view(*[1 for _ in range(len(self.reshape))])  # expand single sigma
                else:
                    logsigma = logsigma.view(*self.reshape)  # reshape feature-specific logsigmas

            return mu, logsigma

