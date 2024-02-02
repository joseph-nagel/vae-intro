'''Variational autoencoder.'''

import math
from warnings import warn

import torch
import torch.nn as nn
import torch.distributions as dist
from lightning.pytorch import LightningModule

from ..data import get_features


LIKELIHOOD_TYPES = (
    'Bernoulli',
    'ContinuousBernoulli',
    'Gauss',
    'Gaussian',
    'Laplace'
)


class VAE(LightningModule):
    '''
    Variational autoencoder with a Bernoulli, Gaussian or Laplace likelihood.

    Summary
    -------
    A VAE algorithm based on a Bernoulli, Gaussian or Laplace likelihood is implemented.
    The overall architecture is composed of a probabilistic encoder and decoder
    that establish the inference model and generative model, respectively.

    The encoder represents the posterior distribution of the latent variables.
    It predicts the means and (logarithmic) standard deviations of a diagonal Gaussian.

    The decoder realizes a multivariate distribution over the data space
    that can be respectively based on Bernoulli, Gaussian or Laplace density functions.
    In the first case, it predicts the logits (inverse sigmoid) of "success" probabilities.
    In the two latter cases, the decoder predicts the mean and standard deviation.

    Parameters
    ----------
    encoder : PyTorch module
        Encoder model that, for given inputs, predicts means and
        logarithmic standard deviations of the latent variables.
    decoder : PyTorch module
        Decoder model predicting Bernoulli logits or mu and logsigma
        of a Gaussian/Laplace distribution for given latent codes.
    num_samples : int
        Number of MC samples to simulate the ELBO.
    likelihood_type : {'Bernoulli', 'ContinuousBernoulli', 'Gauss', 'Gaussian', 'Laplace'}
        Likelihood function type.
    lr : float
        Initial optimizer learning rate.

    '''

    def __init__(self,
                 encoder,
                 decoder,
                 num_samples=1,
                 likelihood_type='Bernoulli',
                 lr=1e-04):

        super().__init__()

        # set encoder and decoder
        self.encoder = encoder
        self.decoder = decoder

        # set number of MC samples
        self.num_samples = abs(int(num_samples))

        # set likelihood type
        if likelihood_type in LIKELIHOOD_TYPES:
            self.likelihood_type = likelihood_type
        else:
            raise ValueError(f'Unknown likelihood type: {likelihood_type}')

        # set initial learning rate
        self.lr = abs(lr)

        # set initial sampling mode
        self.sample(True)

        # store hyperparams
        self.save_hyperparameters(
            ignore=['encoder', 'decoder'],
            logger=True
        )

    @property
    def sampling(self):
        return self._sampling

    @sampling.setter
    def sampling(self, sample_mode):
        self._sampling = sample_mode

    def sample(self, sample_mode=True):
        '''Set sampling mode.'''
        self.sampling = sample_mode

    def train(self, train_mode=True):
        '''Set training mode.'''

        # set module training mode
        super().train(train_mode)

        # turn on sampling for training
        if train_mode:
            self.sample(True)

        return self

    def encode(self, x):
        '''Encode the input variables.'''
        mu, logsigma = self.encoder(x)
        return mu, logsigma

    def reparametrize(self, mu, logsigma=None):
        '''Sample the latent variables.'''

        # sample around the mean
        if self.sampling:
            sigma = torch.exp(logsigma)
            eps = torch.randn_like(sigma)
            z = mu + sigma * eps

        # return the mean value
        else:
            z = mu

        return z

    def decode(self, z):
        '''Decode the latent variables.'''

        # compute Bernoulli logits
        if self.likelihood_type in ('Bernoulli', 'ContinuousBernoulli'):
            logits = self.decoder(z)
            return logits

        # compute Gaussian/Laplace parameters
        else:
            mu, logsigma = self.decoder(z)
            return mu, logsigma

    def __call__(self, x):
        '''Encode, sample, decode.'''

        # encode inputs
        mu, logsigma = self.encode(x)

        # sample latent variables
        z = self.reparametrize(mu, logsigma)

        # compute Bernoulli probabilities
        if self.likelihood_type in ('Bernoulli', 'ContinuousBernoulli'):
            logits = self.decode(z)
            probs = torch.sigmoid(logits)
            return probs

        # compute Gaussian/Laplace parameters
        else:
            mu, logsigma = self.decode(z)
            sigma = torch.exp(logsigma)
            return mu, sigma

    def kl(self, mu, logsigma):
        '''Compute the KL divergence.'''

        sigma = torch.exp(logsigma)
        kl_terms = mu**2 + sigma**2 - 2*logsigma - 1

        # sum over data dimensions (all but batch)
        kl = 0.5 * torch.sum(kl_terms, dim=list(range(1, kl_terms.ndim)))

        return kl

    def ll(self,
           x,
           logits=None,
           mu=None,
           logsigma=None):
        '''Compute the log-likelihood.'''

        # compute Bernoulli likelihood
        if self.likelihood_type == 'Bernoulli':
            if mu is not None or logsigma is not None:
                warn('Gaussian/Laplace parameters are ignored for the Bernoulli log-likelihood')

            # strictly restrict to {0,1}-valued targets (standard Bernoulli)
            if not torch.is_floating_point(x):
                ll_terms = dist.Bernoulli(logits=logits).log_prob(x.float())

            # also allow for [0,1]-valued targets (normalization of the cont. Bernoulli is ignored)
            else:
                ll_terms = -nn.functional.binary_cross_entropy_with_logits(
                    input=logits,
                    target=x,
                    reduction='none'
                )

        # compute continuous Bernoulli likelihood (properly normalized)
        elif self.likelihood_type == 'ContinuousBernoulli':
            if mu is not None or logsigma is not None:
                warn('Gaussian/Laplace parameters are ignored for the (continuous) Bernoulli log-likelihood')

            ll_terms = dist.ContinuousBernoulli(logits=logits).log_prob(x)

        # compute Gaussian likelihood
        elif self.likelihood_type in ('Gauss', 'Gaussian'):
            if logits is not None:
                warn('Bernoulli logits are ignored for the Gaussian log-likelihood')

            sigma = torch.exp(logsigma)
            ll_terms = dist.Normal(loc=mu, scale=sigma).log_prob(x)

        # compute Laplace likelihood
        else:
            if logits is not None:
                warn('Bernoulli logits are ignored for the Laplace log-likelihood')

            sigma = torch.exp(logsigma)
            scale = sigma / math.sqrt(2)
            ll_terms = dist.Laplace(loc=mu, scale=scale).log_prob(x)

        # sum over data dimensions (all but batch)
        ll = torch.sum(ll_terms, dim=list(range(1, ll_terms.ndim)))

        return ll

    def elbo(self, x, num_samples=1):
        '''Estimate the ELBO objective.'''

        # encode inputs
        mu, logsigma = self.encode(x)

        # calculate KL divergence
        kl = self.kl(mu, logsigma)

        # simulate log-likelihood
        ll = torch.zeros_like(kl)

        for _ in range(num_samples):
            z_sample = self.reparametrize(mu, logsigma) # sample latent variables

            # compute sample log-likelihood
            if self.likelihood_type in ('Bernoulli', 'ContinuousBernoulli'):
                logits = self.decode(z_sample)
                ll_sample = self.ll(x, logits=logits)
            else:
                mu, logsigma = self.decode(z_sample)
                ll_sample = self.ll(x, mu=mu, logsigma=logsigma)

            ll = ll + ll_sample # sum log-likelihood samples

        ll = ll / num_samples # compute avarage over samples

        # compute mean over data points (only batch dimension)
        elbo = torch.mean(ll - kl)

        return elbo

    def loss(self, x):
        '''Estimate the negative-ELBO loss.'''
        loss = -self.elbo(x, num_samples=self.num_samples)
        return loss

    def training_step(self, batch, batch_idx):
        x_batch = get_features(batch)
        loss = self.loss(x_batch)
        self.log('train_loss', loss.item()) # Lightning logs batch-wise metrics during training per default
        return loss

    def validation_step(self, batch, batch_idx):
        x_batch = get_features(batch)
        loss = self.loss(x_batch)
        self.log('val_loss', loss.item()) # Lightning automatically averages metrics over batches for validation
        return loss

    def test_step(self, batch, batch_idx):
        x_batch = get_features(batch)
        loss = self.loss(x_batch)
        self.log('test_loss', loss.item()) # Lightning automatically averages metrics over batches for testing
        return loss

    # TODO: enable LR scheduling
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

