'''
Variational autoencoder.

Modules
-------
base : Variational autoencoder.
bernoulli : Bernoulli VAE.
gauss : Gaussian VAE.
utils : Utilities.

'''

from . import base
from . import bernoulli
from . import gauss
from . import utils


from .base import VAE

from .bernoulli import DenseBernoulliVAE, ConvBernoulliVAE

from .gauss import ConvGaussianVAE

from .utils import (
    generate,
    reconstruct,
    encode_loader
)

