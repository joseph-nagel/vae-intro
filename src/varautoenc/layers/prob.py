'''Probabilistic layers.'''

from typing import Any
from collections.abc import Sequence

import torch
import torch.nn as nn

from .utils import make_dense
from .conv import SingleConv, DoubleConv


# define type alias
SigmaType = torch.Tensor | Sequence[float]


class ProbDense(nn.Module):
    '''
    Probabilistic dense layer with constant sigma (fixed or learnable).

    Parameters
    ----------
    in_features : int
        Number of inputs
    out_features : int
        Number of outputs.
    sigma : float or None
        Can be used to specify constant sigmas.
    per_feature : bool
        Enables feature-specific sigma parameters.

    '''

    def __init__(
        self,
        in_features: int,
        out_features: int,
        sigma: SigmaType | None = None,
        per_feature: bool = False,
        **kwargs: Any
    ) -> None:

        super().__init__()

        # create dense layer predicting mu
        self.mu = make_dense(
            in_features,
            out_features,
            **kwargs
        )

        # create log-sigma parameters
        if sigma is None:
            num_sigmas = out_features if per_feature else 1
            logsigma = torch.randn(num_sigmas)
            self.logsigma = nn.Parameter(logsigma, requires_grad=True)

        else:
            sigma = torch.as_tensor(sigma, dtype=torch.float32).detach().clone()
            logsigma = sigma.log().view(-1)
            self.logsigma = nn.Parameter(logsigma, requires_grad=False)

    def forward(self, x):
        mu = self.mu(x)
        logsigma = self.logsigma
        return mu, logsigma


class ProbConv(nn.Module):
    '''
    Probabilistic conv. layer with constant sigma (fixed or learnable).

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    double_conv : bool
        Determines whether double conv. blocks are used.
    sigma : float or None
        Can be used to specify constant sigmas.
    per_channel : bool
        Enables channel-specific sigma parameters.

    '''

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        double_conv: bool = False,
        sigma: SigmaType | None = None,
        per_channel: bool = False,
        **kwargs: Any
    ) -> None:

        super().__init__()

        # determine conv type
        ConvType = DoubleConv if double_conv else SingleConv

        # create conv layer predicting mu
        self.mu = ConvType(
            in_channels,
            out_channels,
            **kwargs
        )

        # create log-sigma parameters
        if sigma is None:
            num_sigmas = out_channels if per_channel else 1
            logsigma = torch.randn(num_sigmas, 1, 1)
            self.logsigma = nn.Parameter(logsigma, requires_grad=True)

        else:
            sigma = torch.as_tensor(sigma, dtype=torch.float32).detach().clone()
            logsigma = sigma.log().view(-1, 1, 1)
            self.logsigma = nn.Parameter(logsigma, requires_grad=False)

    def forward(self, x):
        mu = self.mu(x)
        logsigma = self.logsigma
        return mu, logsigma

