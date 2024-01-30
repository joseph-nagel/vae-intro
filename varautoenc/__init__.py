'''
Variational autoencoder.

Summary
-------
A small playground for experimenting with VAE algorithms is established.
It provides prototypical model implementations with likelihoods that are
either based on a Bernoulli, Gaussian or Laplace probability distribution.
While the former can be applied to {0,1}-valued data (binarized MNIST),
the latter two represent the standards for continuous data (CIFAR-10).

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
    get_features,
    BaseDataModule,
    MNISTDataModule,
    CIFAR10DataModule
)

from .layers import (
    make_activation,
    make_block,
    make_dropout,
    make_dense,
    make_conv,
    make_up,
    DenseBlock,
    MultiDense,
    ProbDense,
    ProbConv,
    SingleConv,
    DoubleConv,
    ConvBlock,
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
    VAE,
    DenseVAE,
    ConvVAE
)

