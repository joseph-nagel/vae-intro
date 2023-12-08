'''Convolutional Bernoulli VAE.'''

from .base import BernoulliVAE
from ..model import ConvEncoder, ConvDecoder


class ConvBernoulliVAE(BernoulliVAE):
    '''Convolutional Bernoulli VAE.'''

    def __init__(self,
                 num_channels,
                 num_features,
                 reshape,
                 kernel_size=3,
                 pooling=2,
                 upsample_mode='conv_transpose',
                 batchnorm=True,
                 activation='leaky_relu',
                 last_activation=None,
                 pool_last=True,
                 lr=1e-04):

        # create encoder
        encoder = ConvEncoder(
            num_channels,
            num_features,
            kernel_size=kernel_size,
            pooling=pooling,
            batchnorm=batchnorm,
            activation=activation,
            pool_last=pool_last
        )

        # create decoder
        decoder = ConvDecoder(
            num_features[::-1],
            num_channels[::-1] if num_channels is not None else None,
            reshape,
            kernel_size=kernel_size,
            scaling=pooling,
            upsample_mode=upsample_mode,
            batchnorm=batchnorm,
            activation=activation,
            last_activation=last_activation
        )

        # initialize VAE class
        super().__init__(
            encoder,
            decoder,
            lr=lr
        )

        # store hyperparams
        self.save_hyperparameters(logger=True)

