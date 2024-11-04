'''Visualization tools.'''

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image

from .vae import (
    DenseVAE,
    ConvVAE,
    generate,
    encode_loader
)


def make_gif(
    save_file,
    img_dir,
    pattern='**/*.png',
    overwrite=True,
    timesort=True,
    **kwargs
):
    '''
    Load images and create GIF animation.

    Summary
    -------
    The function loads a directory of images
    and transforms them into a GIF animation.

    '''

    save_file = Path(save_file)
    img_dir = Path(img_dir)

    # create output dir (if it does not exist)
    save_dir = save_file.parent

    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)

    # get sorted image files
    img_files = sorted(
        img_dir.glob(pattern),
        key=(lambda f: f.stat().st_mtime) if timesort else None # sort according to creation time
    )

    # load frames
    frames = []
    for img_file in img_files:
        # img = imageio.imread(img_file)
        img = Image.open(img_file)
        frames.append(img)

    # save GIF
    if not save_file.exists() or overwrite:
        # imageio.mimsave(save_file, frames, **kwargs)

        # calculate duration per frame in [ms]
        if 'fps' in kwargs:
            kwargs['duration'] = 1000 / kwargs.pop('fps')

        frames[0].save(
            save_file,
            save_all=True,
            append_images=frames[1:],
            **kwargs
        )
    else:
        raise FileExistsError('File already exists')


def make_lat_imgs(
    save_dir,
    ckpt_dir,
    data_loader,
    pattern='**/*.ckpt',
    figsize=(5, 5),
    xlim=(-5, 5),
    ylim=(-5, 5),
    overwrite=True,
    timesort=True,
    **kwargs
):
    '''
    Load checkpoints and save latent space visualizations.

    Summary
    -------
    This function loads all checkpoints in a directory and saves
    visualizations of the corresponding 2D latent spaces.

    '''

    save_dir = Path(save_dir)
    ckpt_dir = Path(ckpt_dir)

    # create output dir
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)

    # get sorted checkpoint files
    ckpt_files = sorted(
        ckpt_dir.glob(pattern),
        key=(lambda f: f.stat().st_mtime) if timesort else None # sort according to creation time
    )

    # set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # loop over checkpoints
    for ckpt_idx, ckpt_file in enumerate(ckpt_files):

        # import model
        vae = DenseVAE.load_from_checkpoint(ckpt_file)

        vae = vae.eval()
        vae = vae.to(device)

        # encode loader
        z_mu, z_sigma, y = encode_loader(
            vae,
            data_loader,
            return_targets=True
        )

        # create figure
        fig, ax = plt.subplots(figsize=figsize)
        for idx in range(10):
            ax.scatter(
                z_mu[y==idx, 0][::2].numpy(),
                z_mu[y==idx, 1][::2].numpy(),
                color=plt.cm.tab10(idx),
                alpha=0.3,
                edgecolors='none',
                label=f'{idx}'
            )
        ax.set(xlim=xlim, ylim=ylim)
        ax.set_aspect('equal', adjustable='box')
        ax.legend(loc='center right')
        ax.grid(visible=True, which='both', color='lightgray', linestyle='-')
        ax.set_axisbelow(True)
        fig.tight_layout()

        # save figure
        file_name = '{}.png'.format(ckpt_file.stem)
        save_file = save_dir / file_name

        if not save_file.exists() or overwrite:
            fig.savefig(save_file, **kwargs)
        else:
            raise FileExistsError('File already exists')

        plt.close(fig)


def make_gen_imgs(
    save_dir,
    ckpt_dir,
    num_latents=None,
    pattern='**/*.ckpt',
    random_seed=None,
    nrows=5,
    ncols=5,
    figsize=(5, 5.5),
    overwrite=True,
    timesort=True,
    **kwargs
):
    '''
    Load checkpoints and save generative visualizations.

    Summary
    -------
    This function reads a directory of checkpoints and saves
    visualizations of generated data from the corresponding models.

    '''

    save_dir = Path(save_dir)
    ckpt_dir = Path(ckpt_dir)

    # create output dir
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)

    # get sorted checkpoint files
    ckpt_files = sorted(
        ckpt_dir.glob(pattern),
        key=(lambda f: f.stat().st_mtime) if timesort else None # sort according to creation time
    )

    # get number of latent variables
    if num_latents is None:
        ckpt_file = ckpt_files[0]
        vae = ConvVAE.load_from_checkpoint(ckpt_file)
        num_latents = vae.decoder.dense_layers[0][0].in_features # TODO: generalize to different architectures

    # set random seed manually
    if random_seed is not None:
        _ = torch.manual_seed(random_seed)

    # generate (fixed) latent samples
    num_samples = nrows * ncols
    sample_shape = (num_latents,)
    z_samples = torch.randn(num_samples, *sample_shape)

    # set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # loop over checkpoints
    for ckpt_idx, ckpt_file in enumerate(ckpt_files):

        # import model
        vae = ConvVAE.load_from_checkpoint(ckpt_file)

        vae = vae.eval()
        vae = vae.to(device)

        # generate samples
        x_gen = generate(vae, z_samples=z_samples)

        # create figure
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        for ax_idx, ax in enumerate(axes.ravel()):
            image = x_gen[ax_idx, 0].numpy().clip(0, 1) # TODO: Generalize to RGB channels
            ax.imshow(image, cmap='gray', vmin=0, vmax=1)
            ax.set(xticks=[], yticks=[], xlabel='', ylabel='')
        fig.suptitle(ckpt_file.stem)
        fig.tight_layout()

        # save figure
        file_name = '{}.png'.format(ckpt_file.stem)
        save_file = save_dir / file_name

        if not save_file.exists() or overwrite:
            fig.savefig(save_file, **kwargs)
        else:
            raise FileExistsError('File already exists')

        plt.close(fig)

