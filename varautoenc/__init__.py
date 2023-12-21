'''
Variational autoencoder.

Modules
-------
data : Data tools.
layers : Model layers.
model : Encoder and decoder models.
vae : Variational autoencoder.

'''

from . import data
from . import layers
from . import model
from . import vae


from .data import (
    BaseDataModule,
    MNISTDataModule
)

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

from .model import (
    DenseEncoder,
    DenseDecoder,
    ConvEncoder,
    ConvDecoder
)

from .vae import (
    generate,
    reconstruct,
    encode_loader,
    BernoulliVAE,
    ConvBernoulliVAE,
    DenseBernoulliVAE
)

