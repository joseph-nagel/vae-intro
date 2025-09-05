'''
Encoder and decoder models.

Modules
-------
conv : Convolutional encoder/decoder.
dense : Dense encoder/decoder.

'''

from . import conv, dense

from .conv import ConvEncoder, ConvDecoder

from .dense import DenseEncoder, DenseDecoder
