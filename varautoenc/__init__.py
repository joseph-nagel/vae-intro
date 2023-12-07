'''
Variational autoencoder.

Modules
-------
layers : Model layers.
vae : Variational autoencoder.

'''

from . import layers
from . import vae


from .layers import (
    make_activation,
    make_block,
    make_dense,
    make_conv,
    make_up,
    DenseModel,
    MultiDense,
    ConvDown,
    ConvUp
)

from .vae import (
    BernoulliVAE,
    ConvEncoder,
    ConvDecoder,
    DenseEncoder,
    DenseDecoder,
    encode_loader
)

