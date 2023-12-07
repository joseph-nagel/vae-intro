'''Variational autoencoder.'''

import torch
import torch.distributions as dist


class BernoulliVAE():
    '''
    Variational autoencoder with Bernoulli likelihood.

    Summary
    -------
    This class establishes a Bernoulli VAE scheme.
    It is initialized with an encoder and decoder model.
    The encoder represents the variational distribution
    of the latent variables as a diagonal Gaussian.
    It predicts means and logarithms of standard deviations.
    The decoder realizes a multivariate Bernoulli distribution
    over the data space, which also defines the likelihood.
    It predicts the corresponding logits to that end.

    Parameters
    ----------
    encoder : PyTorch module
        Encoder model predicting means and
        logarithms of standard deviations.
    decoder : PyTorch module
        Decoder model predicting Bernoulli logits.
    device : PyTorch device
        Device the computations are performed on.

    '''

    def __init__(self, encoder, decoder, device=None):
        self.encoder = encoder
        self.decoder = decoder

        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)

        self.sampling = True
        self.epoch = 0

    def sample(self, sample_mode=True):
        '''Set sampling mode.'''
        self.sampling = sample_mode

    def train(self, train_mode=True):
        '''Set training mode.'''
        self.encoder.train(train_mode)
        self.decoder.train(train_mode)

        if train_mode:
            self.sample(True)

    def encode(self, X):
        '''Encode the input variables.'''
        mu, logsigma = self.encoder(X)
        return mu, logsigma

    def reparametrize(self, mu, logsigma):
        '''Sample the latent variables.'''
        if self.sampling:
            sigma = torch.exp(logsigma)
            eps = torch.randn_like(sigma)
            z = mu + sigma * eps
        else:
            z = mu
        return z

    def decode(self, z):
        '''Decode the latent variables.'''
        logits = self.decoder(z)
        return logits

    def __call__(self, X):
        '''Encode, sample, decode.'''
        mu, logsigma = self.encode(X)
        z = self.reparametrize(mu, logsigma)
        logits = self.decode(z)
        probs = torch.sigmoid(logits)
        return probs

    def kl(self, mu, logsigma):
        '''Compute the KL divergence.'''
        kl = 0.5 * torch.sum(
            mu**2 + torch.exp(logsigma)**2 - 2*logsigma - 1,
            dim=[_ for _ in range(1, mu.ndim)] # sum over data dimensions (all but batch)
        )
        return kl

    def ll(self, x_logits, x):
        '''Compute the log-likelihood.'''
        # ll = torch.sum(
        #     -nn.functional.binary_cross_entropy_with_logits(X_logits, X, reduction='none'),
        #     dim=[_ for _ in range(1, X.ndim)]
        # )
        ll = torch.sum(
            dist.Bernoulli(logits=x_logits).log_prob(x.float()),
            dim=[_ for _ in range(1, x.ndim)] # sum over data dimensions (all but batch)
        )
        return ll

    def elbo(self, x, num_samples=1):
        '''Estimate the ELBO objective.'''

        # run encoder
        mu, logsigma = self.encode(x)

        # calculate KL divergence
        kl = self.kl(mu, logsigma)

        # simulate log-likelihood
        ll = torch.zeros_like(kl)

        for _ in range(num_samples):
            z_sample = self.reparametrize(mu, logsigma) # sample
            x_logits = self.decode(z_sample) # run decoder

            ll = ll + self.ll(x_logits, x)

        ll = ll / num_samples

        # compute ELBO
        elbo = torch.mean(ll - kl) # mean over data points (only batch dimension)

        return elbo

    def loss(self, X, num_samples=1):
        '''Estimate the negative-ELBO loss.'''
        loss = -self.elbo(X, num_samples)
        return loss

    def compile(self, optimizer, train_loader, test_loader=None):
        '''Compile for training.'''
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader

    def fit(self,
            num_epochs,
            num_samples=1,
            log_interval=None,
            initial_test=True):
        '''Perform a number of training epochs.'''

        train_losses = []
        test_losses = []

        if initial_test:
            train_loss = self.test_loss(self.train_loader, num_samples, all_batches=False)
            test_loss = self.test_loss(self.test_loader, num_samples, all_batches=False)

            train_losses.append(train_loss)
            test_losses.append(test_loss)

            print('Started training: {}, test loss: {:.4e}'.format(self.epoch, test_loss))

        for epoch_idx in range(num_epochs):
            train_loss = self.train_epoch(num_samples, log_interval)
            train_losses.append(train_loss)

            self.epoch += 1

            if self.test_loader is not None:
                test_loss = self.test_loss(num_samples=num_samples, all_batches=False)
                test_losses.append(test_loss)

                print('Finished epoch: {}, test loss: {:.4e}'.format(self.epoch, test_loss))

        history = {
            'num_epochs': num_epochs,
            'train_loss': train_losses,
            'test_loss': test_losses
        }

        return history

    def train_epoch(self, num_samples=1, log_interval=None):
        '''Perform a single training epoch.'''

        self.sample(True)
        self.train(True)

        batch_losses = []
        for batch_idx, (x_batch, _) in enumerate(self.train_loader):
            x_batch = x_batch.to(self.device)

            self.optimizer.zero_grad()

            loss = self.loss(x_batch, num_samples=num_samples)

            loss.backward()
            self.optimizer.step()

            batch_loss = loss.data.item()
            batch_losses.append(batch_loss)

            if len(batch_losses) < 3:
                running_loss = batch_loss
            else:
                running_loss = sum(batch_losses[-3:]) / 3

            if log_interval is not None:
                if (batch_idx+1) % log_interval == 0 or (batch_idx+1) == len(self.train_loader):
                    print('Epoch: {} ({}/{}), batch loss: {:.4e}, running loss: {:.4e}' \
                          .format(self.epoch+1, batch_idx+1, len(self.train_loader), batch_loss, running_loss))

        return running_loss

    @torch.no_grad()
    def test_loss(self,
                  test_loader=None,
                  num_samples=1,
                  all_batches=False):
        '''Compute loss over a test set.'''

        if test_loader is None:
            test_loader = self.test_loader

        self.sample(True)
        self.train(False)

        if all_batches: # all batches
            test_loss = 0.0
            for x_batch, _ in test_loader:
                x_batch = x_batch.to(self.device)
                loss = self.loss(x_batch, num_samples=num_samples)
                test_loss += loss.data.item()
        else: # only one batch
            x_batch, _ = next(iter(test_loader))
            x_batch = x_batch.to(self.device)
            loss = self.loss(x_batch, num_samples=num_samples)
            test_loss = loss.data.item()

        return test_loss

