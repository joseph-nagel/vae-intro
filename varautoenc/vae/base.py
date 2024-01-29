'''Variational autoencoder.'''

from warnings import warn

import torch
import torch.distributions as dist
from lightning.pytorch import LightningModule

from ..data import get_features


class VAE(LightningModule):
    '''
    Variational autoencoder with a Bernoulli or Gaussian likelihood.

    Summary
    -------
    A VAE algorithm based on a Bernoulli or Gaussian likelihood is implemented.
    The overall architecture is composed of a probabilistic encoder and decoder
    that establish the inference model and generative model, respectively.

    The encoder represents the posterior distribution of the latent variables.
    It predicts the means and (logarithmic) standard deviations of a diagonal Gaussian.

    The decoder realizes a multivariate Bernoulli distribution over the data space.
    To that end, it predicts the logits (inverse sigmoid) of "success" probabilities.

    Parameters
    ----------
    encoder : PyTorch module
        Encoder model that, for given inputs, predicts means and
        logarithmic standard deviations of the latent variables.
    decoder : PyTorch module
        Decoder model predicting Bernoulli logits or Gaussian
        parameters for given values of the latent variables.
    num_samples : int
        Number of MC samples to simulate the ELBO.
    likelihood_type : {'Bernoulli', 'Gauss', 'Gaussian'}
        Determines the type of the likelihood.
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
        if likelihood_type in ('Bernoulli', 'Gauss', 'Gaussian'):
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
        if self.likelihood_type == 'Bernoulli':
            logits = self.decoder(z)
            return logits

        # compute Gaussian parameters
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
        if self.likelihood_type == 'Bernoulli':
            logits = self.decode(z)
            probs = torch.sigmoid(logits)
            return probs

        # compute Gaussian parameters
        else:
            mu, logsigma = self.decode(z)
            sigma = torch.exp(logsigma)
            return mu, sigma

    def kl(self, mu, logsigma):
        '''Compute the KL divergence.'''
        kl = 0.5 * torch.sum(
            mu**2 + torch.exp(logsigma)**2 - 2*logsigma - 1,
            dim=list(range(1, mu.ndim)) # sum over data dimensions (all but batch)
        )
        return kl

    def ll(self,
           x,
           bernoulli_logits=None,
           gaussian_mu=None,
           gaussian_logsigma=None):
        '''Compute the log-likelihood.'''

        # compute Bernoulli likelihood
        if self.likelihood_type == 'Bernoulli':
            if gaussian_mu is not None or gaussian_logsigma is not None:
                warn('Gaussian parameters are ignored for the Bernoulli log-likelihood')

            ll_terms = dist.Bernoulli(logits=bernoulli_logits).log_prob(x.float())

        # compute Gaussian likelihood
        else:
            if bernoulli_logits is not None:
                warn('Bernoulli logits are ignored for the Gaussian log-likelihood')

            gaussian_sigma = torch.exp(gaussian_logsigma)
            ll_terms = dist.Normal(loc=gaussian_mu, scale=gaussian_sigma).log_prob(x)

        # sum over data dimensions (all but batch)
        ll = torch.sum(ll_terms, dim=list(range(1, x.ndim)))

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
            if self.likelihood_type == 'Bernoulli':
                logits = self.decode(z_sample)
                ll_sample = self.ll(x, bernoulli_logits=logits)
            else:
                mu, logsigma = self.decode(z_sample)
                ll_sample = self.ll(x, gaussian_mu=mu, gaussian_logsigma=logsigma)

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

