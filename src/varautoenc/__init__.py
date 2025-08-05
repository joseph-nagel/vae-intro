'''
Variational autoencoder.

Summary
-------
A small playground for experimenting with VAE algorithms is established.
It provides prototypical model implementations with likelihoods that can be
based on (continuous) Bernoulli, Gaussian or Laplace probability distributions.

Strictly speaking, the standard Bernoulli applies to {0, 1}-valued data only,
while its continuous variant represents an option for [0, 1]-valued data.
Gauss or Laplace distributions are applicable to continuous data more generally.

Modules
-------
data : Data tools.
layers : Model layers.
model : Encoder and decoder models.
vae : Variational autoencoder.
vis : Visualization tools.

'''

from . import (
    data,
    layers,
    model,
    vae,
    vis
)

from .data import (
    get_batch,
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
    LIKELIHOOD_TYPES,
    make_lr_schedule,
    generate,
    reconstruct,
    encode_loader,
    VAE,
    DenseVAE,
    ConvVAE
)

from .vis import (
    make_gif,
    make_lat_imgs,
    make_gen_imgs
)

