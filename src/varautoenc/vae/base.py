'''Variational autoencoder.'''

from typing import Self

import math

import torch
import torch.nn as nn
import torch.distributions as dist
from lightning.pytorch import LightningModule

from ..data import BatchType, get_features
from .lr_schedule import make_lr_schedule


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
    beta : float
        Beta-VAE weighting parameter.
    num_samples : int
        Number of MC samples to simulate the ELBO.
    likelihood_type : {'Bernoulli', 'ContinuousBernoulli', 'Gauss', 'Gaussian', 'Laplace'}
        Likelihood function type.
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
        encoder: nn.Module,
        decoder: nn.Module,
        beta: float = 1.0,
        num_samples: int = 1,
        likelihood_type: str = 'Bernoulli',
        lr: float = 1e-04,
        lr_schedule: str | None = 'constant',
        lr_interval: str = 'epoch',
        lr_warmup: int = 0
    ) -> None:

        super().__init__()

        # set encoder and decoder
        self.encoder = encoder
        self.decoder = decoder

        # set beta weight
        if beta is None:
            beta = 1.0

        self.beta = abs(beta)

        # set number of MC samples
        self.num_samples = abs(int(num_samples))

        # set likelihood type
        if likelihood_type in LIKELIHOOD_TYPES:
            self.likelihood_type = likelihood_type
        else:
            raise ValueError(f'Unknown likelihood type: {likelihood_type}')

        # set LR params
        self.lr = abs(lr)
        self.lr_schedule = lr_schedule
        self.lr_interval = lr_interval
        self.lr_warmup = abs(int(lr_warmup))

        # set initial sampling mode
        self.sample(True)

        # store hyperparams
        self.save_hyperparameters(
            ignore=['encoder', 'decoder'],
            logger=True
        )

    @property
    def sampling(self) -> bool:
        return self._sampling

    @sampling.setter
    def sampling(self, sample_mode: bool) -> None:
        self._sampling = sample_mode

    def sample(self, sample_mode: bool = True) -> None:
        '''Set sampling mode.'''
        self.sampling = sample_mode

    def train(self, train_mode: bool = True) -> Self:
        '''Set training mode.'''

        # set module training mode
        super().train(train_mode)

        # turn on sampling for training
        if train_mode:
            self.sample(True)

        return self

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        '''Encode the input variables.'''
        # mu, logsigma = self.encoder(x)
        return self.encoder(x)

    def reparametrize(self, mu: torch.Tensor, logsigma: torch.Tensor) -> torch.Tensor:
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

    def decode(self, z: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        '''Decode the latent variables.'''
        return self.decoder(z)

        # if self.likelihood_type in ('Bernoulli', 'ContinuousBernoulli'):
        #     logits = self.decoder(z)
        # else:
        #     mu, logsigma = self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        '''Encode, sample, decode.'''

        # encode inputs
        z_mu, z_logsigma = self.encode(x)

        # sample latent variables
        z = self.reparametrize(z_mu, z_logsigma)

        # decode latents
        dist_params = self.decode(z)

        # compute Bernoulli probabilities
        if self.likelihood_type in ('Bernoulli', 'ContinuousBernoulli'):

            if not isinstance(dist_params, torch.Tensor):
                raise TypeError(f'Invalid input type {type(dist_params)} for Bernoulli distribution')

            logits = dist_params
            probs = torch.sigmoid(logits)

            return probs

        # compute Gaussian/Laplace parameters
        else:

            if not isinstance(dist_params, (tuple, list)):
                raise TypeError(f'Invalid input type {type(dist_params)} for Gauss/Laplace distribution')

            elif len(dist_params) != 2:
                raise ValueError(f'Invalid input length {len(dist_params)} for Gauss/Laplace distribution')

            mu, logsigma = dist_params
            sigma = torch.exp(logsigma)

            return mu, sigma

    def kl(self, mu: torch.Tensor, logsigma: torch.Tensor) -> torch.Tensor:
        '''Compute the KL divergence.'''

        sigma = torch.exp(logsigma)
        kl_terms = mu**2 + sigma**2 - 2*logsigma - 1

        # sum over data dimensions (all but batch)
        kl = 0.5 * torch.sum(kl_terms, dim=list(range(1, kl_terms.ndim)))

        return kl

    def ll(
        self,
        x: torch.Tensor,
        dist_params: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        '''Compute the log-likelihood.'''

        # compute Bernoulli log-likelihood
        if self.likelihood_type in ('Bernoulli', 'ContinuousBernoulli'):

            if not isinstance(dist_params, torch.Tensor):
                raise TypeError(f'Invalid input type {type(dist_params)} for Bernoulli likelihood')

            logits = dist_params

            # compute standard Bernoulli likelihood
            if self.likelihood_type == 'Bernoulli':

                # strictly restrict to {0, 1}-valued targets (standard Bernoulli)
                if not torch.is_floating_point(x):
                    ll_terms = dist.Bernoulli(logits=logits).log_prob(x.float())

                # also allow for [0, 1]-valued targets (normalization of the cont. Bernoulli is ignored)
                else:
                    ll_terms = -nn.functional.binary_cross_entropy_with_logits(
                        input=logits,
                        target=x,
                        reduction='none'
                    )

            # compute continuous Bernoulli likelihood (properly normalized)
            elif self.likelihood_type == 'ContinuousBernoulli':
                ll_terms = dist.ContinuousBernoulli(logits=logits).log_prob(x)

        # compute Gaussian/Laplace log-likelihood
        else:

            if not isinstance(dist_params, (tuple, list)):
                raise TypeError(f'Invalid input type {type(dist_params)} for Gauss/Laplace likelihood')
            elif len(dist_params) != 2:
                raise ValueError(f'Invalid input length {len(dist_params)} for Gauss/Laplace likelihood')

            mu, logsigma = dist_params

            # compute Gaussian likelihood
            if self.likelihood_type in ('Gauss', 'Gaussian'):
                sigma = torch.exp(logsigma)
                ll_terms = dist.Normal(loc=mu, scale=sigma).log_prob(x)

            # compute Laplace likelihood
            else:
                sigma = torch.exp(logsigma)
                scale = sigma / math.sqrt(2)
                ll_terms = dist.Laplace(loc=mu, scale=scale).log_prob(x)

        # sum over data dimensions (all but batch)
        ll = torch.sum(ll_terms, dim=list(range(1, ll_terms.ndim)))

        return ll

    def elbo(self, x: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        '''Estimate the ELBO objective.'''

        # encode inputs
        z_mu, z_logsigma = self.encode(x)

        # calculate KL divergence
        kl = self.kl(z_mu, z_logsigma)

        # simulate log-likelihood
        ll = torch.zeros_like(kl)

        for _ in range(num_samples):
            z_sample = self.reparametrize(z_mu, z_logsigma)  # sample latent variables

            # compute sample log-likelihood
            ll_sample = self.ll(x, dist_params=self.decode(z_sample))

            ll = ll + ll_sample  # sum log-likelihood samples

        ll = ll / num_samples  # compute avarage over samples

        # compute mean over data points (only batch dimension)
        elbo = torch.mean(ll - self.beta * kl)

        return elbo

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        '''Estimate the negative-ELBO loss.'''
        loss = -self.elbo(x, num_samples=self.num_samples)
        return loss

    def training_step(self, batch: BatchType, batch_idx: int) -> torch.Tensor:
        x_batch = get_features(batch)
        loss = self.loss(x_batch)
        self.log('train_loss', loss.item())  # Lightning logs batch-wise scalars during training per default
        return loss

    def validation_step(self, batch: BatchType, batch_idx: int) -> torch.Tensor:
        x_batch = get_features(batch)
        loss = self.loss(x_batch)
        self.log('val_loss', loss.item())  # Lightning automatically averages scalars over batches for validation
        return loss

    def test_step(self, batch: BatchType, batch_idx: int) -> torch.Tensor:
        x_batch = get_features(batch)
        loss = self.loss(x_batch)
        self.log('test_loss', loss.item())  # Lightning automatically averages scalars over batches for testing
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer | tuple[list, list]:

        # create optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        # return optimizer only (if no LR schedule has been set)
        if self.lr_schedule is None:
            return optimizer

        # create LR schedule (if a schedule has been set)
        else:

            # get total number of training time units
            if self.lr_interval == 'epoch':
                num_total = self.trainer.max_epochs
            elif self.lr_interval == 'step':
                num_total = self.trainer.estimated_stepping_batches
            else:
                raise ValueError(f'Unknown LR interval: {self.lr_interval}')

            # create LR scheduler
            lr_scheduler = make_lr_schedule(
                optimizer=optimizer,
                mode=self.lr_schedule,
                num_total=num_total,
                num_warmup=self.lr_warmup,
                last_epoch=-1
            )

            # create LR config
            lr_config = {
                'scheduler': lr_scheduler,  # set LR scheduler
                'interval': self.lr_interval,  # set time unit (step or epoch)
                'frequency': 1  # set update frequency
            }

            return [optimizer], [lr_config]
