'''Variational autoencoder.'''

import torch
import torch.distributions as dist
from lightning.pytorch import LightningModule


class BernoulliVAE(LightningModule):
    '''
    Variational autoencoder with Bernoulli likelihood.

    Summary
    -------
    A VAE algorithm based on a Bernoulli likelihood is implemented.
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
        Decoder model predicting Bernoulli logits
        for given values of the latent variables.
    lr : float
        Initial optimizer learning rate.

    '''

    def __init__(self,
                 encoder,
                 decoder,
                 lr=1e-04):

        super().__init__()

        # set encoder and decoder
        self.encoder = encoder
        self.decoder = decoder

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
        logits = self.decoder(z)
        return logits

    def __call__(self, x):
        '''Encode, sample, decode.'''

        # encode inputs
        mu, logsigma = self.encode(x)

        # sample latent variables
        z = self.reparametrize(mu, logsigma)

        # decode logits
        logits = self.decode(z)

        # compute probabilities
        probs = torch.sigmoid(logits)

        return probs

    def kl(self, mu, logsigma):
        '''Compute the KL divergence.'''
        kl = 0.5 * torch.sum(
            mu**2 + torch.exp(logsigma)**2 - 2*logsigma - 1,
            dim=list(range(1, mu.ndim)) # sum over data dimensions (all but batch)
        )
        return kl

    def ll(self, x_logits, x):
        '''Compute the log-likelihood.'''
        # ll = torch.sum(
        #     -nn.functional.binary_cross_entropy_with_logits(x_logits, x, reduction='none'),
        #     dim=list(range(1, x.ndim))
        # )
        ll = torch.sum(
            dist.Bernoulli(logits=x_logits).log_prob(x.float()),
            dim=list(range(1, x.ndim)) # sum over data dimensions (all but batch)
        )
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
            x_logits = self.decode(z_sample) # decode logits

            ll = ll + self.ll(x_logits, x) # sum log-likelihood samples

        ll = ll / num_samples # compute avarage over samples

        # compute ELBO
        elbo = torch.mean(ll - kl) # mean over data points (only batch dimension)

        return elbo

    def loss(self, x, num_samples=1):
        '''Estimate the negative-ELBO loss.'''
        loss = -self.elbo(x, num_samples)
        return loss

    @staticmethod
    def _get_features(batch):
        '''Get only batch features and discard the rest.'''

        if isinstance(batch, torch.Tensor):
            x_batch = batch

        elif isinstance(batch, (tuple, list)):
            x_batch = batch[0]

        elif isinstance(batch, dict):
            x_batch = batch['features']

        else:
            raise TypeError(f'Invalid batch type: {type(batch)}')

        return x_batch

    def training_step(self, batch, batch_idx):
        x_batch = self._get_features(batch)
        loss = self.loss(x_batch)
        self.log('train_loss', loss.item()) # Lightning logs batch-wise metrics during training per default
        return loss

    def validation_step(self, batch, batch_idx):
        x_batch = self._get_features(batch)
        loss = self.loss(x_batch)
        self.log('val_loss', loss.item()) # Lightning automatically averages metrics over batches for validation
        return loss

    def test_step(self, batch, batch_idx):
        x_batch = self._get_features(batch)
        loss = self.loss(x_batch)
        self.log('test_loss', loss.item()) # Lightning automatically averages metrics over batches for testing
        return loss

    # TODO: enable LR scheduling
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

