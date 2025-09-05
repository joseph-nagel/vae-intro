'''
Generated images animation.

Example
-------
python scripts/make_gen_gif.py --save-dir run/anim_gen/ --ckpt-dir run/mnist_conv/version_0/checkpoints/ --random-seed 100000

'''

from argparse import ArgumentParser
from pathlib import Path

from varautoenc import make_gif, make_gen_imgs


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--random-seed', type=int, required=False, help='Random seed')

    parser.add_argument('--save-dir', type=Path, required=True, help='Output directory')
    parser.add_argument('--ckpt-dir', type=Path, required=True, help='Checkpoint directory')
    parser.add_argument('--pattern', type=str, default='**/epoch=*.ckpt', help='Filename pattern')

    parser.add_argument('--num-latents', type=int, required=False, help='Number of latent variables')

    parser.add_argument('--nrows', type=int, default=5, help='Number of figure rows')
    parser.add_argument('--ncols', type=int, default=5, help='Number of figures columns')
    parser.add_argument('--figsize', type=int, nargs='+', default=[5, 5.5], help='Figsize specification')

    parser.add_argument('--dpi', type=int, default=120, help='Dots per inch')

    parser.add_argument('--loop', type=int, default=0, help='Number of loops (0 is for infinite)')
    parser.add_argument('--fps', type=float, default=5.0, help='Frames per second')

    parser.add_argument('--overwrite', dest='overwrite', action='store_true', help='Overwrite existing files')
    parser.add_argument('--no-overwrite', dest='overwrite', action='store_false', help='Do not overwrite')
    parser.set_defaults(overwrite=True)

    args = parser.parse_args()

    return args


def main(args):

    # create and save images
    make_gen_imgs(
        save_dir=args.save_dir,
        ckpt_dir=args.ckpt_dir,
        num_latents=args.num_latents,
        pattern=args.pattern,
        random_seed=args.random_seed,
        nrows=args.nrows,
        ncols=args.ncols,
        figsize=args.figsize,
        overwrite=args.overwrite,
        timesort=True,
        dpi=args.dpi
    )

    # create GIF from images
    make_gif(
        save_file=args.save_dir / 'anim.gif',
        img_dir=args.save_dir,
        pattern='**/*.png',
        overwrite=args.overwrite,
        timesort=True,
        fps=args.fps,
        loop=args.loop
    )


if __name__ == '__main__':

    args = parse_args()
    main(args)
