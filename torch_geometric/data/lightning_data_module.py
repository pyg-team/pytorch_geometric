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
    r"""Converts a :class:`torch_geometric.data.Dataset` into a
    :class:`pytorch_lightning.LightningDataModule` variant, which can be
    automatically used as a :obj:`data_module` for multi-GPU training via
    `PyTorch Lightning <https://www.pytorchlightning.ai>`_.
    :class:`LightningDataset` will take care of providing mini-batches via
    :class:`torch_geometric.loader.DataLoader`.

    .. note::

        Currently only supports the :obj:`"ddp_spawn"` training strategy of
        PyTorch Lightning:

        .. code-block::

            import pytorch_lightning as pl
            trainer = pl.Trainer(strategy="ddp_spawn')

    Args:
        train_dataset: (Dataset) The training dataset.
        val_dataset: (Dataset) The validation dataset.
        test_dataset: (Dataset, optional) The test dataset.
            (default: :obj:`None`)
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        num_workers: How many subprocesses to use for data loading.
            :obj:`0` means that the data will be loaded in the main process.
            (default: :obj:`0`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric..loader.DataLoader`.
    """
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
                          f"'{self.__class__.__name__}'. Remove it from the "
                          f"argument list to disable this warning")
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
            raise NotImplementedError(
                f"'{self.__class__.__name__}' currently only supports the "
                f"'ddp_spawn' training strategy of 'pytorch_lightning'")

    def dataloader(self, dataset_name: str, shuffle: bool) -> DataLoader:
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
        from torch.utils.data import IterableDataset
        shuffle = not isinstance(self.train_dataset, IterableDataset)
        return self.dataloader('train_dataset', shuffle=shuffle)

    def val_dataloader(self) -> DataLoader:
        return self.dataloader('val_dataset', shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self.dataloader('test_dataset', shuffle=False)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'train_dataset={self.train_dataset}, '
                f'val_dataset={self.val_dataset}, '
                f'test_dataset={self.test_dataset}, '
                f'batch_size={self.batch_size})')
