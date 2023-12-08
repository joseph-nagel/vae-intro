'''Dense Bernoulli VAE.'''

from .base import BernoulliVAE
from ..model import DenseEncoder, DenseDecoder


class DenseBernoulliVAE(BernoulliVAE):
    '''Fully connected Bernoulli VAE.'''

    def __init__(self,
                 num_features,
                 reshape,
                 activation='leaky_relu',
                 lr=1e-04):

        # create encoder
        encoder = DenseEncoder(
            num_features,
            activation=activation,
            flatten=True,
        )

        # create decoder
        decoder = DenseDecoder(
            num_features[::-1],
            activation=activation,
            reshape=reshape
        )

        # initialize VAE class
        super().__init__(
            encoder,
            decoder,
            lr=lr
        )

        # store hyperparams
        self.save_hyperparameters(logger=True)

