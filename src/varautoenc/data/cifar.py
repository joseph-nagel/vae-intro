'''CIFAR-10 datamodule.'''

import torch
from torch.utils.data import random_split
from torchvision import datasets, transforms

from .base import BaseDataModule
from .utils import FloatOrFloats


class CIFAR10DataModule(BaseDataModule):
    '''DataModule for the CIFAR-10 dataset.'''

    def __init__(
        self,
        data_dir: str,
        mean: FloatOrFloats | None = (0.5, 0.5, 0.5),
        std: FloatOrFloats | None = (0.5, 0.5, 0.5),
        batch_size: int = 32,
        num_workers: int = 0
    ):

        # call base class init
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers
        )

        # set data location
        self.data_dir = data_dir

        # create transforms
        transforms_list = [transforms.ToTensor()]

        if (mean is not None) and (std is not None):  # normalize (e.g. scale to [-1, 1])
            normalize_fn = transforms.Normalize(mean=mean, std=std)
            transforms_list.append(normalize_fn)

        self.transform = transforms.Compose(transforms_list)

        # create inverse normalization
        if (mean is not None) and (std is not None):

            mean = torch.as_tensor(mean).view(-1, 1, 1)
            std = torch.as_tensor(std).view(-1, 1, 1)

            self.renormalize = transforms.Compose([
                transforms.Lambda(lambda x: x * std + mean),  # reverse normalization
                transforms.Lambda(lambda x: x.clamp(0, 1))  # clip to valid range
            ])

    def prepare_data(self) -> None:
        '''Download data.'''
        train_set = datasets.CIFAR10(
            self.data_dir,
            train=True,
            download=True
        )
        test_set = datasets.CIFAR10(
            self.data_dir,
            train=False,
            download=True
        )

    def setup(self, stage: str) -> None:
        '''Set up train/test/val. datasets.'''

        # create train/val. datasets
        if stage in ('fit', 'validate'):
            train_set = datasets.CIFAR10(
                self.data_dir,
                train=True,
                transform=self.transform
            )

            self.train_set, self.val_set = random_split(
                train_set,
                [40000, 10000],
                generator=torch.Generator().manual_seed(42)
            )

        # create test dataset
        elif stage == 'test':
            self.test_set = datasets.CIFAR10(
                self.data_dir,
                train=False,
                transform=self.transform
            )
