'''Visualization tools.'''

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import imageio

from .vae import ConvVAE, generate


def make_imgs(save_dir,
              ckpt_dir,
              pattern='**/*.ckpt',
              num_latents=None,
              random_seed=None,
              nrows=5,
              ncols=5,
              figsize=(5, 5.5),

              **kwargs):
    '''
    Load checkpoints and save visualizations.

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
    ckpt_files = sorted(ckpt_dir.glob(pattern), key=lambda f: f.stat().st_mtime)

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
        file_name = 'frame_{:03d}.png'.format(ckpt_idx + 1)
        file_path = save_dir / file_name

        fig.savefig(file_path, **kwargs)
        plt.close(fig)


def make_gif(save_file,
             img_dir,
             pattern='**/frame_*.png',
             **kwargs):
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
    img_files = sorted(img_dir.glob(pattern), key=lambda f: f.stat().st_mtime)

    # loop over images
    frames = []
    for img_file in img_files:

        # load frame
        img = imageio.imread(img_file)
        frames.append(img)

    # save GIF
    imageio.mimsave(save_file, frames, **kwargs)

