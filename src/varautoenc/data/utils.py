'''Data utilities.'''

from collections.abc import Sequence

import torch


# define type aliases
FloatOrFloats = float | tuple[float, float, float]
BatchType = torch.Tensor | Sequence[torch.Tensor] | dict[str, torch.Tensor]


def get_batch(batch: BatchType) -> tuple[torch.Tensor, torch.Tensor]:
    '''Get batch features and labels.'''

    if isinstance(batch, (tuple, list)):
        x_batch = batch[0]
        y_batch = batch[1]

    elif isinstance(batch, dict):
        x_batch = batch['features']
        y_batch = batch['labels']

    else:
        raise TypeError(f'Invalid batch type: {type(batch)}')

    return x_batch, y_batch


def get_features(batch: BatchType) -> torch.Tensor:
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
