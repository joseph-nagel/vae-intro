'''Dense layers.'''

from collections.abc import Sequence

import torch
import torch.nn as nn

from .utils import ActivType, make_dense


class DenseBlock(nn.Sequential):
    '''Multiple (serial) dense layers.'''

    def __init__(
        self,
        num_features: Sequence[int],
        batchnorm: bool = False,
        activation: ActivType | None = 'leaky_relu',
        last_activation: ActivType | None = 'same',
        normalize_last: bool = True,
        drop_rate: float | None = None
    ) -> None:

        # determine last activation
        if last_activation == 'same':
            last_activation = activation

        # check number of layers
        if len(num_features) >= 2:
            num_layers = len(num_features) - 1
        else:
            raise ValueError('Number of features needs at least two entries')

        # assemble layers
        layers = []
        for idx, (in_features, out_features) in enumerate(zip(num_features[:-1], num_features[1:])):
            is_not_last = (idx < num_layers - 1)

            dense = make_dense(
                in_features,
                out_features,
                batchnorm=batchnorm if is_not_last else (batchnorm and normalize_last),
                activation=activation if is_not_last else last_activation,
                drop_rate=drop_rate
            )

            layers.append(dense)

        # initialize module
        super().__init__(*layers)


class MultiDense(nn.Module):
    '''Multiple (parallel) dense layers.'''

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_outputs: int,
        batchnorm: bool = False,
        activation: ActivType | None = None,
        drop_rate: float | None = None
    ) -> None:

        super().__init__()

        layers = [] # type: list[nn.Module]

        for _ in range(num_outputs):

            dense = make_dense(
                in_features,
                out_features,
                batchnorm=batchnorm,
                activation=activation,
                drop_rate=drop_rate
            )

            layers.append(dense)

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return [l(x) for l in self.layers]

