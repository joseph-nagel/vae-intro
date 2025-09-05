'''Base datamodule.'''

from torch.utils.data import Dataset, DataLoader
from lightning import LightningDataModule


class BaseDataModule(LightningDataModule):
    '''DataModule base class.'''

    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 0
    ) -> None:

        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_set = None
        self.val_set = None
        self.test_set = None

    def set_dataset(self, data_set: Dataset, mode: str) -> None:
        '''Set dataset.'''

        if mode in ('train', 'val', 'test'):
            setattr(self, mode + '_set', data_set)
        else:
            raise ValueError(f'Unknown dataset mode: {mode}')

    def train_dataloader(self) -> DataLoader:
        '''Create train dataloader.'''

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

    def val_dataloader(self) -> DataLoader:
        '''Create val. dataloader.'''

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

    def test_dataloader(self) -> DataLoader:
        '''Create test dataloader.'''

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
