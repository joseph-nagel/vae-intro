'''MNIST datamodule.'''

import torch
from torch.utils.data import random_split
from torchvision import datasets, transforms

from .base import BaseDataModule


class MNISTDataModule(BaseDataModule):
    '''DataModule for the (binarized) MNIST dataset.'''

    def __init__(
        self,
        data_dir: str,
        binarize_threshold: float | None = None,
        mean: float | None = None,
        std: float | None = None,
        batch_size: int = 32,
        num_workers: int = 0
    ) -> None:

        # call base class init
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers
        )

        # set data location
        self.data_dir = data_dir

        # create transforms
        train_transforms = [
            transforms.RandomRotation(5),  # TODO: refine data augmentation
            transforms.ToTensor()
        ]

        test_transforms = [transforms.ToTensor()]

        if binarize_threshold is not None:  # binarize to {0, 1}
            binarize_fn = lambda x: torch.where(x > binarize_threshold, 1, 0).float()

            train_transforms.append(binarize_fn)
            test_transforms.append(binarize_fn)

        if (mean is not None) and (std is not None):  # normalize (e.g. scale to [-1, 1])
            normalize_fn = transforms.Normalize(mean=mean, std=std)

            train_transforms.append(normalize_fn)
            test_transforms.append(normalize_fn)

        self.train_transform = transforms.Compose(train_transforms)
        self.test_transform = transforms.Compose(test_transforms)

        # create inverse normalization
        if (mean is not None) and (std is not None):

            self.renormalize = transforms.Compose([
                transforms.Lambda(lambda x: x * std + mean),  # reverse normalization
                transforms.Lambda(lambda x: x.clamp(0, 1))  # clip to valid range
            ])

    def prepare_data(self) -> None:
        '''Download data.'''

        train_set = datasets.MNIST(
            self.data_dir,
            train=True,
            download=True
        )

        test_set = datasets.MNIST(
            self.data_dir,
            train=False,
            download=True
        )

    def setup(self, stage: str) -> None:
        '''Set up train/test/val. datasets.'''

        # create train/val. datasets
        if stage in ('fit', 'validate'):
            train_set = datasets.MNIST(
                self.data_dir,
                train=True,
                transform=self.train_transform
            )

            self.train_set, self.val_set = random_split(
                train_set,
                [50000, 10000],
                generator=torch.Generator().manual_seed(42)
            )

        # create test dataset
        elif stage == 'test':
            self.test_set = datasets.MNIST(
                self.data_dir,
                train=False,
                transform=self.test_transform
            )

