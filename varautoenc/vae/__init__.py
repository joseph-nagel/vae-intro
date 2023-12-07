'''Variational autoencoder.'''

from . import base
from . import conv
from . import dense
from . import utils


from .base import BernoulliVAE

from .conv import ConvEncoder, ConvDecoder

from .dense import DenseEncoder, DenseDecoder

from .utils import encode_loader

