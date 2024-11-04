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
    batchnorm : bool
        Determines whether batchnorm is used.
    activation : str or None
        Nonlinearity type.
    drop_rate : float or None
        Dropout probability for dense layers.
    beta : float
        Beta-VAE weighting parameter.
    num_samples : int
        Number of MC samples to simulate the ELBO.
    likelihood_type : {'Bernoulli', 'ContinuousBernoulli', 'Gauss', 'Gaussian', 'Laplace'}
        Likelihood function type.
    sigma : float or None
        Can be used to specify a constant sigma.
    per_feature : bool
        Enables feature-specific sigma parameters.
    lr : float
        Initial optimizer learning rate.

    '''

    def __init__(
        self,
        num_features: Sequence[int],
        reshape: Sequence[int] | None = None,
        batchnorm: bool = False,
        activation: ActivType | None = 'leaky_relu',
        drop_rate: float | None = None,
        beta: float = 1.0,
        num_samples: int = 1,
        likelihood_type: str = 'Bernoulli',
        sigma: SigmaType | None = None,
        per_feature: bool = False,
        lr: float = 0.001
    ) -> None:

        # create encoder (predicts Gaussian params)
        encoder = DenseEncoder(
            num_features,
            batchnorm=batchnorm,
            activation=activation,
            drop_rate=drop_rate,
            flatten=True # flatten the input
        )

        # create decoder (predicts Bernoulli logits or Gaussian params)
        decoder = DenseDecoder(
            num_features[::-1],
            batchnorm=batchnorm,
            activation=activation,
            drop_rate=drop_rate,
            reshape=reshape, # reshape the flat output
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
            lr=lr
        )

        # store hyperparams
        self.save_hyperparameters(logger=True)

