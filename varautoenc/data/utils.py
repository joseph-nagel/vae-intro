'''Data utilities.'''

import torch


def get_features(batch):
    '''Get only batch features and discard the rest.'''

    if isinstance(batch, torch.Tensor):
        x_batch = batch

    elif isinstance(batch, (tuple, list)):
        x_batch = batch[0]

    elif isinstance(batch, dict):
        x_batch = batch['features']

    else:
        raise TypeError(f'Invalid batch type: {type(batch)}')

    return x_batch

