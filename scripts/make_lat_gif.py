'''
Latent space animation.

Example
-------
python scripts/make_lat_gif.py --save-dir run/anim_lat/ --ckpt-dir run/mnist_dense/version_0/checkpoints/

'''

from argparse import ArgumentParser
from pathlib import Path

from varautoenc import MNISTDataModule, make_gif, make_lat_imgs


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--data-dir', type=Path, default='run/data/', help='Data dir')

    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=0, help='Number of workers')

    parser.add_argument('--binarize', dest='binarize', action='store_true', help='Binarize MNIST data')
    parser.add_argument('--no-binarize', dest='binarize', action='store_false', help='Do not binarize MNIST data')
    parser.set_defaults(binarize=False)

    parser.add_argument('--save-dir', type=Path, required=True, help='Output directory')
    parser.add_argument('--ckpt-dir', type=Path, required=True, help='Checkpoint directory')
    parser.add_argument('--pattern', type=str, default='**/step=*.ckpt', help='Filename pattern')

    parser.add_argument('--figsize', type=int, nargs='+', default=[5, 5], help='Figsize specification')
    parser.add_argument('--xlim', type=int, nargs='+', default=[-4.5, 5.5], help='X-axis limits')
    parser.add_argument('--ylim', type=int, nargs='+', default=[-5, 5], help='Y-axis limits')

    parser.add_argument('--dpi', type=int, default=80, help='Dots per inch')

    parser.add_argument('--loop', type=int, default=0, help='Number of loops (0 is for infinite)')
    parser.add_argument('--fps', type=float, default=20.0, help='Frames per second')

    parser.add_argument('--overwrite', dest='overwrite', action='store_true', help='Overwrite existing files')
    parser.add_argument('--no-overwrite', dest='overwrite', action='store_false', help='Do not overwrite')
    parser.set_defaults(overwrite=True)

    args = parser.parse_args()

    return args


def main(args):

    # create dataloader
    mnist = MNISTDataModule(
        data_dir=args.data_dir,
        binarize_threshold=0.5 if args.binarize else None,
        mean=None,
        std=None,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    mnist.prepare_data()
    mnist.setup(stage='test')

    test_loader = mnist.test_dataloader()

    # create and save images
    make_lat_imgs(
        save_dir=args.save_dir,
        ckpt_dir=args.ckpt_dir,
        data_loader=test_loader,
        pattern=args.pattern,
        figsize=args.figsize,
        xlim=args.xlim,
        ylim=args.ylim,
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
