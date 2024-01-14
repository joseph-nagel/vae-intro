'''Gaussian VAE.'''

from .base import VAE
from ..model import ConvEncoder, ProbDecoder


class ConvGaussianVAE(VAE):
    '''Convolutional Gaussian VAE.'''

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
                 sigma=None,
                 per_channel=False,
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

        # create decoder (predicts Gaussian mu and logsigma)
        decoder = ProbDecoder(
            num_features[::-1],
            num_channels[::-1] if num_channels is not None else None,
            reshape,
            kernel_size=kernel_size,
            scaling=pooling,
            upsample_mode=upsample_mode,
            batchnorm=batchnorm,
            activation=activation,
            last_activation=last_activation,
            sigma=sigma,
            per_channel=per_channel
        )

        # initialize VAE class
        super().__init__(
            encoder,
            decoder,
            num_samples=num_samples,
            likelihood_type='Gaussian',
            lr=lr
        )

        # store hyperparams
        self.save_hyperparameters(logger=True)

