'''
Gaussian VAE training on CIFAR-10.

Example
-------
python scripts/train_gaussian_vae.py --num-channels 3 16 32 --num-features 2048 512 128 --reshape 32 8 8

'''

from argparse import ArgumentParser
from pathlib import Path

import torch
from lightning.pytorch import seed_everything, Trainer
from lightning.pytorch.loggers import TensorBoardLogger, MLFlowLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    StochasticWeightAveraging
)

from varautoenc import CIFAR10DataModule, ConvVAE


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--random-seed', type=int, required=False, help='Random seed')

    parser.add_argument('--ckpt-file', type=Path, required=False, help='Checkpoint for resuming')

    parser.add_argument('--logger', type=str, default='tensorboard', help='Logger')
    parser.add_argument('--save-dir', type=Path, default='run/', help='Save dir')
    parser.add_argument('--name', type=str, default='cifar10', help='Experiment name')
    parser.add_argument('--version', type=str, required=False, help='Experiment version')

    parser.add_argument('--data-dir', type=Path, default='run/data/', help='Data dir')

    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=0, help='Number of workers')

    parser.add_argument('--num-channels', type=int, nargs='+', required=True, help='Channel numbers of conv. layers')
    parser.add_argument('--num-features', type=int, nargs='+', required=True, help='Feature numbers of linear layers')
    parser.add_argument('--reshape', type=int, nargs='+', required=True, help='Shape between linear and conv. layers')
    parser.add_argument('--kernel-size', type=int, default=3, help='Conv. kernel size')
    parser.add_argument('--pooling', type=int, default=2, help='Pooling parameter')
    parser.add_argument('--upsample-mode', type=str, default='conv_transpose', help='Conv. upsampling mode')
    parser.add_argument('--activation', type=str, default='leaky_relu', help='Nonlinearity type')
    parser.add_argument('--drop-rate', type=float, required=False, help='Dropout probability for dense layers')

    parser.add_argument('--batchnorm', dest='batchnorm', action='store_true', help='Use batchnorm for conv. layers')
    parser.add_argument('--no-batchnorm', dest='batchnorm', action='store_false', help='Do not use batchnorm for conv. layers')
    parser.set_defaults(batchnorm=True)

    parser.add_argument('--pool-last', dest='pool_last', action='store_true', help='Pool after last conv.')
    parser.add_argument('--no-pool-last', dest='pool_last', action='store_false', help='Do not pool after last conv.')
    parser.set_defaults(pool_last=True)

    parser.add_argument('--double-conv', dest='double_conv', action='store_true', help='Use double conv. blocks')
    parser.add_argument('--single-conv', dest='double_conv', action='store_false', help='Use single convolutions')
    parser.set_defaults(double_conv=True)

    parser.add_argument('--per-channel', dest='per_channel', action='store_true', help='Use channel-specific sigmas')
    parser.add_argument('--same-sigma', dest='per_channel', action='store_false', help='Use same sigma for all channels')
    parser.set_defaults(per_channel=False)

    parser.add_argument('--num-samples', type=int, default=1, help='Number of MC samples')

    parser.add_argument('--lr', type=float, default=1e-04, help='Optimizer learning rate')

    parser.add_argument('--max-epochs', type=int, default=20, help='Max. number of training epochs')

    parser.add_argument('--save-top', type=int, default=1, help='Number of best models to save')
    parser.add_argument('--save-every', type=int, default=1, help='Regular checkpointing interval')

    parser.add_argument('--patience', type=int, default=0, help='Early stopping patience')

    parser.add_argument('--swa-lrs', type=float, default=0.0, help='SWA learning rate')
    parser.add_argument('--swa-epoch-start', type=float, default=0.7, help='SWA start epoch')
    parser.add_argument('--annealing-epochs', type=int, default=10, help='SWA annealing epochs')
    parser.add_argument('--annealing-strategy', type=str, default='cos', help='SWA annealing strategy')

    parser.add_argument('--gradient-clip-val', type=float, default=0.0, help='Gradient clipping value')
    parser.add_argument('--gradient-clip-algorithm', type=str, default='norm', help='Gradient clipping mode')

    parser.add_argument('--gpu', dest='gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--cpu', dest='gpu', action='store_false', help='Do not use GPU')
    parser.set_defaults(gpu=True)

    args = parser.parse_args()

    return args


def main(args):

    # set random seeds
    if args.random_seed is not None:
        _ = seed_everything(
            args.random_seed,
            workers=args.num_workers > 0
        )

    # initialize datamodule
    cifar = CIFAR10DataModule(
        data_dir=args.data_dir,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # initialize model
    vae = ConvVAE(
        num_channels=args.num_channels,
        num_features=args.num_features,
        reshape=args.reshape,
        kernel_size=args.kernel_size,
        pooling=args.pooling,
        upsample_mode=args.upsample_mode,
        batchnorm=args.batchnorm,
        activation=args.activation,
        last_activation=None,
        drop_rate=args.drop_rate,
        pool_last=args.pool_last,
        double_conv=args.double_conv,
        num_samples=args.num_samples,
        likelihood_type='Gaussian',
        sigma=None,
        per_channel=args.per_channel,
        lr=args.lr
    )

    # set accelerator
    if args.gpu:
        accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    else:
        accelerator = 'cpu'

    # create logger
    if args.logger == 'tensorboard':
        logger = TensorBoardLogger(
            args.save_dir,
            name=args.name,
            version=args.version
        )
    elif args.logger == 'mlflow':
        logger = MLFlowLogger(
            experiment_name=args.name,
            run_name=args.version,
            save_dir=args.save_dir / 'mlruns',
            log_model=True
        )
    else:
        raise ValueError('Unknown logger: {}'.format(args.logger))

    # set up checkpointing
    save_top_ckpt = ModelCheckpoint(
        filename='best',
        monitor='val_loss',
        mode='min',
        save_top_k=args.save_top,
    )

    save_every_ckpt = ModelCheckpoint(
        filename='{epoch}_{val_loss:.2f}',
        save_top_k=-1,
        every_n_epochs=args.save_every,
        save_last=True
    )

    callbacks = [save_top_ckpt, save_every_ckpt]

    # set up early stopping
    if args.patience > 0:
        early_stopping = EarlyStopping('val_loss', patience=args.patience)
        callbacks.append(early_stopping)

    # set up weight averaging
    if args.swa_lrs > 0:
        swa = StochasticWeightAveraging(
            swa_lrs=args.swa_lrs,
            swa_epoch_start=args.swa_epoch_start,
            annealing_epochs=args.annealing_epochs,
            annealing_strategy=args.annealing_strategy
        )
        callbacks.append(swa)

    # set up gradient clipping
    if args.gradient_clip_val > 0:
        gradient_clip_val = args.gradient_clip_val
        gradient_clip_algorithm = args.gradient_clip_algorithm
    else:
        gradient_clip_val = None
        gradient_clip_algorithm = None

    # initialize trainer
    trainer = Trainer(
        accelerator=accelerator,
        devices=1,
        logger=logger,
        callbacks=callbacks,
        max_epochs=args.max_epochs,
        log_every_n_steps=100,
        gradient_clip_val=gradient_clip_val,
        gradient_clip_algorithm=gradient_clip_algorithm,
        deterministic=args.random_seed is not None
    )

    # check validation loss
    trainer.validate(
        model=vae,
        datamodule=cifar,
        ckpt_path=args.ckpt_file,
        verbose=False
    )

    # train model
    trainer.fit(
        vae,
        datamodule=cifar,
        ckpt_path=args.ckpt_file
    )


if __name__ == '__main__':

    args = parse_args()
    main(args)

