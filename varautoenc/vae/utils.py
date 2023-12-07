'''Utilities.'''

import torch


@torch.no_grad()
def encode_loader(vae, data_loader):
    '''Encode all items in a data loader.'''

    vae.sample(False)
    vae.train(False)

    y = []
    z_mu = []
    z_sigma = []

    for x_batch, y_batch in data_loader:
        y.append(y_batch)

        x_batch = x_batch.to(vae.device)
        y_batch = y_batch.to(vae.device)

        z_batch_mu, z_batch_logsigma = vae.encode(x_batch)
        z_batch_sigma = torch.exp(z_batch_logsigma)

        z_mu.append(z_batch_mu)
        z_sigma.append(z_batch_sigma)

    y = torch.cat(y, dim=0)
    z_mu = torch.cat(z_mu, dim=0)
    z_sigma = torch.cat(z_sigma, dim=0)

    return z_mu, z_sigma, y

