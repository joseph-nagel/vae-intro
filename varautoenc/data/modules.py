'''Datamodules.'''

import torch
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms
from lightning import LightningDataModule


class BaseDataModule(LightningDataModule):
    '''DataModule base class.'''

    def __init__(self, batch_size=32, num_workers=0):

        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_set = None
        self.val_set = None
        self.test_set = None

    def set_dataset(self, mode, data_set):
        '''Set dataset.'''
        if mode in ('train', 'val', 'test'):
            setattr(self, mode + '_set', data_set)
        else:
            raise ValueError(f'Unknown dataset mode: {mode}')

    def train_dataloader(self):
        if hasattr(self, 'train_set') and self.train_set is not None:
            return DataLoader(
                self.train_set,
                batch_size=self.batch_size,
                drop_last=True,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.num_workers > 0
            )
        else:
            raise AttributeError('Train set has not been set')

    def val_dataloader(self):
        if hasattr(self, 'val_set') and self.val_set is not None:
            return DataLoader(
                self.val_set,
                batch_size=self.batch_size,
                drop_last=False,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.num_workers > 0
            )
        else:
            raise AttributeError('Val. set has not been set')

    def test_dataloader(self):
        if hasattr(self, 'test_set') and self.test_set is not None:
            return DataLoader(
                self.test_set,
                batch_size=self.batch_size,
                drop_last=False,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.num_workers > 0
            )
        else:
            raise AttributeError('Test set has not been set')


class MNISTDataModule(BaseDataModule):
    '''DataModule for the (binarized) MNIST dataset.'''

    def __init__(self,
                 data_dir,
                 binarize_threshold=0.5,
                 batch_size=32,
                 num_workers=0):

        # call base class init
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers
        )

        # set data location
        self.data_dir = data_dir

        # create transforms
        train_transforms = [
            transforms.RandomRotation(5), # TODO: refine data augmentation
            transforms.ToTensor()
        ]

        test_transforms = [transforms.ToTensor()]

        if binarize_threshold is not None:
            binarize_fn = lambda x: torch.where(x > binarize_threshold, 1, 0).float()

            train_transforms.append(binarize_fn)
            test_transforms.append(binarize_fn)

        self.train_transform = transforms.Compose(train_transforms)
        self.test_transform = transforms.Compose(test_transforms)

    def prepare_data(self):
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

    def setup(self, stage):
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


class CIFAR10DataModule(BaseDataModule):
    '''DataModule for the CIFAR-10 dataset.'''

    def __init__(self,
                 data_dir,
                 batch_size=32,
                 num_workers=0):

        # call base class init
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers
        )

        # set data location
        self.data_dir = data_dir

        # create transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5)
            )
        ])

    def prepare_data(self):
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

    def setup(self, stage):
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

