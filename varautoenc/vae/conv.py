'''Convolutional VAE.'''

from .base import VAE
from ..model import ConvEncoder, ConvDecoder


class ConvVAE(VAE):
    '''
    Convolutional VAE.

    Parameters
    ----------
    num_channels : list
        Channel numbers for conv. layers.
    num_features : list
        Feature numbers for dense layers.
    reshape : list
        Shape between dense and conv. layers.
    kernel_size : int
        Conv. kernel size.
    pooling : int
        Pooling parameter.
    upsample_mode : {'bilinear', 'bilinear_conv', 'conv_transpose'}
        Conv. upsampling mode.
    batchnorm : bool
        Determines whether batchnorm is used.
    activation : str
        Nonlinearity type.
    last_activation : str
        Nonlinearity of the final layer.
    drop_rate : float
        Dropout probability for dense layers.
    pool_last : bool
        Controls the last pooling operation (also first upscaling).
    double_conv : bool
        Determines whether double conv. blocks are used.
    beta : float
        Beta-VAE weighting parameter.
    num_samples : int
        Number of MC samples to simulate the ELBO.
    likelihood_type : {'Bernoulli', 'ContinuousBernoulli', 'Gauss', 'Gaussian', 'Laplace'}
        Likelihood function type.
    sigma : float
        Can be used to specify a constant sigma.
    per_channel : bool
        Enables channel-specific sigma parameters.
    lr : float
        Initial optimizer learning rate.

    '''

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
                 drop_rate=None,
                 pool_last=True,
                 double_conv=True,
                 beta=1.0,
                 num_samples=1,
                 likelihood_type='Bernoulli',
                 sigma=None,
                 per_channel=False,
                 lr=0.001):

        # create encoder (predicts Gaussian params)
        encoder = ConvEncoder(
            num_channels,
            num_features,
            kernel_size=kernel_size,
            pooling=pooling,
            batchnorm=batchnorm,
            activation=activation,
            drop_rate=drop_rate,
            pool_last=pool_last,
            double_conv=double_conv,
            inout_first=True
        )

        # create decoder (predicts Bernoulli logits or Gaussian params)
        decoder = ConvDecoder(
            num_features[::-1],
            num_channels[::-1] if num_channels is not None else None,
            reshape,
            kernel_size=kernel_size,
            scaling=pooling,
            upsample_mode=upsample_mode,
            batchnorm=batchnorm,
            activation=activation,
            last_activation=last_activation,
            drop_rate=drop_rate,
            up_first=pool_last,
            double_conv=double_conv,
            inout_first=True,
            likelihood_type=likelihood_type,
            sigma=sigma,
            per_channel=per_channel
        )

        # initialize VAE class
        super().__init__(
            encoder,
            decoder,
            beta=beta,
            num_samples=num_samples,
            likelihood_type=likelihood_type,
            lr=lr
        )

        # store hyperparams
        self.save_hyperparameters(logger=True)

