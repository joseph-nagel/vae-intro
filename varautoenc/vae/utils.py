'''Utilities.'''

import torch


@torch.no_grad()
def generate(vae, sample_shape, num_samples=1):
    '''Generate random samples.'''

    vae.sample(False) # actually not necessary
    vae.train(False) # activate train mode

    # sample latent variables
    z_samples = torch.randn(num_samples, *sample_shape, device=vae.device)

    # run decoder
    x_gen = vae.decode(z_samples)
    x_gen = x_gen.cpu()

    return x_gen


@torch.no_grad()
def reconstruct(vae, x, sample_mode=False):
    '''Reconstruct inputs.'''

    vae.sample(sample_mode) # set sampling mode
    vae.train(False) # activate train mode

    # run encoder and decoder
    x_recon = vae(x.to(vae.device))
    x_recon = x_recon.cpu()

    return x_recon


@torch.no_grad()
def encode_loader(vae, data_loader, return_inputs=False):
    '''Encode all items in a data loader.'''

    vae.sample(False) # actually not necessary
    vae.train(False) # activate train mode

    z_mu = []
    z_sigma = []

    if return_inputs:
        y = []

    # loop over batches
    for x_batch, y_batch in data_loader:

        if return_inputs:
            y.append(y_batch)

        # run encoder
        z_batch_mu, z_batch_logsigma = vae.encode(x_batch.to(vae.device))

        z_batch_mu = z_batch_mu.cpu()
        z_batch_logsigma = z_batch_logsigma.cpu()

        z_batch_sigma = torch.exp(z_batch_logsigma)

        z_mu.append(z_batch_mu)
        z_sigma.append(z_batch_sigma)

    z_mu = torch.cat(z_mu, dim=0)
    z_sigma = torch.cat(z_sigma, dim=0)

    if return_inputs:
        y = torch.cat(y, dim=0)

    if return_inputs:
        return z_mu, z_sigma, y
    else:
        return z_mu, z_sigma

