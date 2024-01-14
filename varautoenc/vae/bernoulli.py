'''Bernoulli VAE.'''

from .base import VAE
from ..model import (
    DenseEncoder,
    DenseDecoder,
    ConvEncoder,
    ConvDecoder
)


class DenseBernoulliVAE(VAE):
    '''Fully connected Bernoulli VAE.'''

    def __init__(self,
                 num_features,
                 reshape,
                 activation='leaky_relu',
                 num_samples=1,
                 lr=1e-04):

        # create encoder (predicts Gaussian mu and logsigma)
        encoder = DenseEncoder(
            num_features,
            activation=activation,
            flatten=True,
        )

        # create decoder (predicts Bernoulli logits)
        decoder = DenseDecoder(
            num_features[::-1],
            activation=activation,
            reshape=reshape
        )

        # initialize VAE class
        super().__init__(
            encoder,
            decoder,
            num_samples=num_samples,
            likelihood_type='Bernoulli',
            lr=lr
        )

        # store hyperparams
        self.save_hyperparameters(logger=True)


class ConvBernoulliVAE(VAE):
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
                 num_samples=1,
                 lr=1e-04):

        # create encoder (predicts Gaussian mu and logsigma)
        encoder = ConvEncoder(
            num_channels,
            num_features,
            kernel_size=kernel_size,
            pooling=pooling,
            batchnorm=batchnorm,
            activation=activation,
            pool_last=pool_last
        )

        # create decoder (predicts Bernoulli logits)
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
            num_samples=num_samples,
            likelihood_type='Bernoulli',
            lr=lr
        )

        # store hyperparams
        self.save_hyperparameters(logger=True)

