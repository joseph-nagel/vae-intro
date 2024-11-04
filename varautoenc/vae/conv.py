'''Convolutional VAE.'''

from collections.abc import Sequence

from .base import VAE
from ..model import ConvEncoder, ConvDecoder
from ..layers import IntOrInts, ActivType, SigmaType


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
    kernel_size : int or (int, int)
        Conv. kernel size.
    pooling : int, (int, int) or None
        Pooling parameter.
    upsample_mode : {'bilinear', 'bilinear_conv', 'conv_transpose'}
        Conv. upsampling mode.
    batchnorm : bool
        Determines whether batchnorm is used.
    activation : str or None
        Nonlinearity type.
    last_activation : str or None
        Nonlinearity of the final layer.
    drop_rate : float or None
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
    sigma : float or None
        Can be used to specify a constant sigma.
    per_channel : bool
        Enables channel-specific sigma parameters.
    lr : float
        Initial optimizer learning rate.

    '''

    def __init__(
        self,
        num_channels: Sequence[int],
        num_features: Sequence[int],
        reshape: Sequence[int],
        kernel_size: IntOrInts = 3,
        pooling: IntOrInts | None = 2,
        upsample_mode: str = 'conv_transpose',
        batchnorm: bool = True,
        activation: ActivType | None = 'leaky_relu',
        last_activation: ActivType | None = None,
        drop_rate: float | None = None,
        pool_last: bool = True,
        double_conv: bool = True,
        beta: float = 1.0,
        num_samples: int = 1,
        likelihood_type: str = 'Bernoulli',
        sigma: SigmaType | None = None,
        per_channel: bool = False,
        lr: float = 0.001
    ) -> None:

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

