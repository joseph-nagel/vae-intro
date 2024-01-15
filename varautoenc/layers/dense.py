'''Dense layers.'''

import torch.nn as nn

from .utils import make_dense


class DenseModel(nn.Sequential):
    '''Fully connected model.'''

    def __init__(self,
                 num_features,
                 activation='leaky_relu',
                 last_activation='same',
                 drop_rate=None):

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
                activation=activation if is_not_last else last_activation,
                drop_rate=drop_rate
            )

            layers.append(dense)

        # initialize module
        super().__init__(*layers)


class MultiDense(nn.Module):
    '''Multiple dense layers.'''

    def __init__(self,
                 in_features,
                 out_features,
                 num_outputs,
                 activation=None,
                 drop_rate=None):

        super().__init__()

        layers = []
        for _ in range(num_outputs):

            dense = make_dense(
                in_features,
                out_features,
                activation=activation,
                drop_rate=drop_rate
            )

            layers.append(dense)

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        return [l(x) for l in self.layers]

