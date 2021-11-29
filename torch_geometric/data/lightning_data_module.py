from typing import Optional

from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

import pytorch_lightning as pl


class LightningDataset(pl.LightningDataModule):
    def __init__(self, train_dataset: Dataset, val_dataset: Dataset,
                 test_dataset: Optional[Dataset] = None, **kwargs):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.kwargs = kwargs

    def prepare_data(self):
        print("PREPARE DATA")

    def setup(self, stage: Optional[str] = None):
        print("SETUP", stage)

    def train_dataloader(self):
        print("TRAIN DATALOADER")
        # TODO: shuffle=True
        # TODO: pin_memory=True?
        loader = DataLoader(self.train_dataset, **self.kwargs)
        print(loader.batch_size)
        return loader

    def val_dataloader(self):
        print("VAL DATALOADER")
        # TODO: pin_memory=True?
        loader = DataLoader(self.val_dataset, **self.kwargs)
        print(loader.batch_size)
        return loader

    def test_dataloader(self):
        print("TEST DATALOADER")
        # TODO: pin_memory=True?
        loader = DataLoader(self.test_dataset, **self.kwargs)
        print(loader.batch_size)
        return loader

    # TODO: replicate dataset functionality
