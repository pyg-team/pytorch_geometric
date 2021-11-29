from typing import Optional

import warnings

from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

try:
    from pytorch_lightning import LightningDataModule
    no_pytorch_lightning = False
except (ImportError, ModuleNotFoundError):
    LightningDataModule = object
    no_pytorch_lightning = True


class LightningDataset(LightningDataModule):
    def __init__(self, train_dataset: Dataset, val_dataset: Dataset,
                 test_dataset: Optional[Dataset] = None, batch_size: int = 1,
                 num_workers: int = 0, **kwargs):
        super().__init__()

        if no_pytorch_lightning:
            raise ModuleNotFoundError(
                "No module named 'pytorch_lightning'. Please install it "
                "via 'pip install pytorch_lightning'")

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.kwargs = kwargs

        if 'shuffle' in kwargs:
            shuffle = kwargs['shuffle']
            warnings.warn(f"The 'shuffle={shuffle}' option is ignored in "
                          f"'{self.__class__.__name__}'")
            del kwargs['shuffle']

        if 'pin_memory' in kwargs:
            self.pin_memory = kwargs['pin_memory']
            del kwargs['pin_memory']
        else:
            self.pin_memory = True

        if 'persistent_workers' in kwargs:
            self.persistent_workers = kwargs['persistent_workers']
            del kwargs['persistent_workers']
        else:
            self.persistent_workers = num_workers > 0

    def setup(self, stage: Optional[str] = None):
        from pytorch_lightning.plugins import DDPSpawnPlugin
        if not isinstance(self.trainer.training_type_plugin, DDPSpawnPlugin):
            raise NotImplementedError

    def dataloader(self, dataset_name: str, shuffle: bool) -> DataLoader:
        from torch.utils.data import IterableDataset
        shuffle &= not isinstance(self.train_dataset, IterableDataset)
        return DataLoader(
            dataset=getattr(self, dataset_name),
            batch_size=self.batch_size,
            shuffle=shuffle,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            **self.kwargs,
        )

    def train_dataloader(self) -> DataLoader:
        return self.dataloader('train_dataset', shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self.dataloader('val_dataset', shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self.dataloader('test_dataset', shuffle=False)
