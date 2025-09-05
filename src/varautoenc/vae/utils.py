'''Utilities.'''

from collections.abc import Sequence

import torch
from torch.utils.data import DataLoader

from .base import VAE


@torch.no_grad()
def generate(
    vae: VAE,
    sample_shape: Sequence[int] | None = None,
    num_samples: int | None = None,
    z_samples: torch.Tensor | None = None,
    random_seed: int | None = None
) -> torch.Tensor:
    '''Generate random samples.'''

    vae.sample(False)  # actually not necessary
    vae.train(False)  # activate train mode

    # set random seed manually
    if random_seed is not None:
        _ = torch.manual_seed(random_seed)

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

    if isinstance(x_gen, torch.Tensor):
        x_gen = torch.sigmoid(x_gen)  # compute probabilities from logits (Bernoulli)
    elif isinstance(x_gen, (tuple, list)):
        if len(x_gen) == 2:
            x_gen = x_gen[0]  # get first entry of a (mu, logsigma)-tuple (Gaussian/Laplace)
        else:
            raise ValueError(f'Two dist. parameters expected, found: {len(x_gen)}')
    else:
        raise TypeError(f'Invalid decoder output type : {type(x_gen)}')

    x_gen = x_gen.cpu()

    return x_gen


@torch.no_grad()
def reconstruct(
    vae: VAE,
    x: torch.Tensor,
    sample_mode: bool = False
) -> torch.Tensor:
    '''Reconstruct inputs.'''

    vae.sample(sample_mode)  # set sampling mode
    vae.train(False)  # activate train mode

    # run encoder and decoder
    x_recon = vae(x.to(vae.device))

    if isinstance(x_recon, (tuple, list)):
        x_recon = x_recon[0]  # get first entry of a (mu, logsigma)-tuple

    x_recon = x_recon.cpu()

    return x_recon


@torch.no_grad()
def encode_loader(
    vae: VAE,
    data_loader: DataLoader,
    return_targets: bool = False
) -> tuple[torch.Tensor, ...]:
    '''Encode all items in a data loader.'''

    vae.sample(False)  # actually not necessary
    vae.train(False)  # activate train mode

    z_mu_list = []
    z_sigma_list = []

    if return_targets:
        y_list = []

    # loop over batches
    for x_batch, y_batch in data_loader:

        if return_targets:
            y_list.append(y_batch)

        # run encoder
        z_batch_mu, z_batch_logsigma = vae.encode(x_batch.to(vae.device))

        z_batch_mu = z_batch_mu.cpu()
        z_batch_logsigma = z_batch_logsigma.cpu()

        z_batch_sigma = torch.exp(z_batch_logsigma)

        z_mu_list.append(z_batch_mu)
        z_sigma_list.append(z_batch_sigma)

    z_mu = torch.cat(z_mu_list, dim=0)
    z_sigma = torch.cat(z_sigma_list, dim=0)

    if return_targets:
        y = torch.cat(y_list, dim=0)

    if return_targets:
        return z_mu, z_sigma, y
    else:
        return z_mu, z_sigma
