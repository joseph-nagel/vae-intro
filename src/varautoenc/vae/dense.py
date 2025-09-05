'''Dense VAE.'''

from collections.abc import Sequence

from .base import VAE
from ..model import DenseEncoder, DenseDecoder
from ..layers import ActivType, SigmaType


class DenseVAE(VAE):
    '''
    Dense VAE.

    Parameters
    ----------
    num_features : list
        Feature numbers for dense layers.
    reshape : list or None
        Final output shape.
    activation : str or None
        Nonlinearity type.
    batchnorm : bool
        Determines whether batchnorm is used.
    drop_rate : float or None
        Dropout probability for dense layers.
    beta : float
        Beta-VAE weighting parameter.
    num_samples : int
        Number of MC samples to simulate the ELBO.
    likelihood_type : {'Bernoulli', 'ContinuousBernoulli', 'Gauss', 'Gaussian', 'Laplace'}
        Likelihood function type.
    sigma : float or None
        Can be used to specify constant sigmas.
    per_feature : bool
        Enables feature-specific sigma parameters.
    lr : float
        Initial learning rate.
    lr_schedule : {'constant', 'cosine'} or None
        Learning rate schedule type.
    lr_interval : {'epoch', 'step'}
        Learning rate update interval.
    lr_warmup : int
        Warmup steps/epochs.

    '''

    def __init__(
        self,
        num_features: Sequence[int],
        reshape: Sequence[int] | None = None,
        activation: ActivType | None = 'leaky_relu',
        batchnorm: bool = False,
        drop_rate: float | None = None,
        beta: float = 1.0,
        num_samples: int = 1,
        likelihood_type: str = 'Bernoulli',
        sigma: SigmaType | None = None,
        per_feature: bool = False,
        lr: float = 0.001,
        lr_schedule: str | None = 'constant',
        lr_interval: str = 'epoch',
        lr_warmup: int = 0
    ) -> None:

        # create encoder (predicts Gaussian params)
        encoder = DenseEncoder(
            num_features,
            activation=activation,
            batchnorm=batchnorm,
            drop_rate=drop_rate,
            flatten=True  # flatten the input
        )

        # create decoder (predicts Bernoulli logits or Gaussian params)
        decoder = DenseDecoder(
            num_features[::-1],
            activation=activation,
            batchnorm=batchnorm,
            drop_rate=drop_rate,
            reshape=reshape,  # reshape the flat output
            likelihood_type=likelihood_type,
            sigma=sigma,
            per_feature=per_feature
        )

        # initialize VAE class
        super().__init__(
            encoder,
            decoder,
            beta=beta,
            num_samples=num_samples,
            likelihood_type=likelihood_type,
            lr=lr,
            lr_schedule=lr_schedule,
            lr_interval=lr_interval,
            lr_warmup=lr_warmup
        )

        # store hyperparams
        self.save_hyperparameters(logger=True)
