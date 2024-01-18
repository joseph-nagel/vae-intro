'''
Variational autoencoder.

Modules
-------
base : Variational autoencoder.
conv : Convolutional VAE.
dense : Dense VAE
utils : Utilities.

'''

from . import base
from . import conv
from . import dense
from . import utils


from .base import VAE

from .conv import ConvVAE

from .dense import DenseVAE

from .utils import (
    generate,
    reconstruct,
    encode_loader
)

