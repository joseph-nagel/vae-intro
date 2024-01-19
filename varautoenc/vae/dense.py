'''Dense VAE.'''

from .base import VAE
from ..model import DenseEncoder, DenseDecoder


class DenseVAE(VAE):
    '''
    Dense VAE.

    Parameters
    ----------
    num_features : list
        Feature numbers for dense layers.
    reshape : list
        Final output shape.
    batchnorm : bool
        Determines whether batchnorm is used.
    activation : str
        Nonlinearity type.
    drop_rate : float
        Dropout probability for dense layers.
    num_samples : int
        Number of MC samples to simulate the ELBO.
    likelihood_type : {'Bernoulli', 'Gauss', 'Gaussian'}
        Likelihood function type.
    sigma : float
        Can be used to specify a constant sigma.
    per_feature : bool
        Enables feature-specific sigma parameters.
    lr : float
        Initial optimizer learning rate.

    '''

    def __init__(self,
                 num_features,
                 reshape=None,
                 batchnorm=False,
                 activation='leaky_relu',
                 drop_rate=None,
                 num_samples=1,
                 likelihood_type='Bernoulli',
                 sigma=None,
                 per_feature=False,
                 lr=0.001):

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
            num_samples=num_samples,
            likelihood_type=likelihood_type,
            lr=lr
        )

        # store hyperparams
        self.save_hyperparameters(logger=True)

