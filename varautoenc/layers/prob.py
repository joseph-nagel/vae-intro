'''Probabilistic layers.'''

import torch
import torch.nn as nn

from .utils import make_dense
from .conv import SingleConv, DoubleConv


class ProbDense(nn.Module):
    '''Probabilistic dense layer with constant sigma (fixed or learnable).'''

    def __init__(self,
                 in_features,
                 out_features,
                 sigma=None,
                 per_feature=False,
                 **kwargs):

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
    '''Probabilistic conv. layer with constant sigma (fixed or learnable).'''

    def __init__(self,
                 in_channels,
                 out_channels,
                 double_conv=False,
                 sigma=None,
                 per_channel=False,
                 **kwargs):

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

