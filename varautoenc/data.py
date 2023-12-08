'''Data tools.'''

import torch
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms
from lightning import LightningDataModule


class BinarizedMNIST(LightningDataModule):
    '''DataModule for the binarized MNIST dataset.'''

    def __init__(self,
                 data_dir,
                 batch_size=32,
                 num_workers=0):

        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        # create transforms
        self.train_transform = transforms.Compose([
            transforms.RandomRotation(5),
            transforms.ToTensor(),
            lambda x: torch.where(x > 0.5, 1, 0).float()
        ]) # TODO: refine data augmentation

        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            lambda x: torch.where(x > 0.5, 1, 0).float()
        ])

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

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

