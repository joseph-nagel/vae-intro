'''Utilities.'''

import torch


@torch.no_grad()
def generate(vae,
             sample_shape=None,
             num_samples=None,
             z_samples=None):
    '''Generate random samples.'''

    vae.sample(False) # actually not necessary
    vae.train(False) # activate train mode

    # sample latent variables
    if z_samples is None:
        if sample_shape is None:
            raise TypeError('A sample shape has to be specified')
        elif num_samples is None:
            raise TypeError('The number of samples has to be specified')
        else:
            z_samples = torch.randn(num_samples, *sample_shape)

    # use passed variables as latents
    else:
        if sample_shape is not None:
            raise TypeError('A sample shape should not be specified')
        elif num_samples is not None:
            raise TypeError('The number of samples should not be specified')
        else:
            z_samples = torch.as_tensor(z_samples)

    # run decoder
    z_samples = z_samples.to(device=vae.device)

    x_gen = vae.decode(z_samples)

    if isinstance(x_gen, (tuple, list)):
        x_gen = x_gen[0] # get first entry of a (mu, logsigma)-tuple

    x_gen = x_gen.cpu()

    return x_gen


@torch.no_grad()
def reconstruct(vae, x, sample_mode=False):
    '''Reconstruct inputs.'''

    vae.sample(sample_mode) # set sampling mode
    vae.train(False) # activate train mode

    # run encoder and decoder
    x_recon = vae(x.to(vae.device))

    if isinstance(x_recon, (tuple, list)):
        x_recon = x_recon[0] # get first entry of a (mu, logsigma)-tuple

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

