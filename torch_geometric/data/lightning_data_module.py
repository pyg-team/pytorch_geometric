from typing import Optional

from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader


def LightningDataset(self, train_dataset: Dataset, val_dataset: Dataset,
                     test_dataset: Optional[Dataset] = None,
                     batch_size: int = 1, num_workers: int = 0, **kwargs):

    from pytorch_lightning import LightningDataModule
    from pytorch_lightning.plugins import DDPSpawnPlugin

    def setup(self, stage: Optional[str] = None):
        if not isinstance(self.trainer.training_type_plugin, DDPSpawnPlugin):
            raise NotImplementedError

    def dataloader(dataset: Dataset, shuffle: bool, **kwargs):
        # shuffle &= not isinstance(dataset, IterableDataset)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                          pin_memory=True, num_workers=num_workers,
                          persistent_workers=num_workers > 0, **kwargs)

    data_module = LightningDataModule()
    data_module.train_dataset = train_dataset
    data_module.val_dataset = val_dataset
    data_module.test_dataset = test_dataset

    data_module.train_dataloader = dataloader
    data_module.val_dataloader = dataloader
    if test_dataset is not None:
        data_module.train_dataloader = dataloader

    return data_module
