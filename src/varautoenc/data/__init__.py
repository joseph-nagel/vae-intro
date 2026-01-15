'''
Data tools.

Modules
-------
base : Base datamodule.
cifar : CIFAR-10 datamodule.
mnist : MNIST datamodule.
utils : Data utilities.

'''

from . import base, cifar, mnist, utils
from .base import BaseDataModule
from .cifar import CIFAR10DataModule
from .mnist import MNISTDataModule
from .utils import (
    FloatOrFloats,
    BatchType,
    get_batch,
    get_features
)
