'''Variational autoencoder.'''

from . import base
from . import conv
from . import dense
from . import utils


from .base import BernoulliVAE

from .conv import ConvBernoulliVAE

from .dense import DenseBernoulliVAE

from .utils import encode_loader

