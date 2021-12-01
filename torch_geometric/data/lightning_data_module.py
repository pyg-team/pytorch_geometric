from typing import Optional

import warnings

from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

try:
    from pytorch_lightning import LightningDataModule as PLLightningDataModule
    no_pytorch_lightning = False
except (ImportError, ModuleNotFoundError):
    PLLightningDataModule = object
    no_pytorch_lightning = True


class LightningDataModule(PLLightningDataModule):
    def __init__(self, has_val: bool, has_test: bool, **kwargs):
        if no_pytorch_lightning:
            raise ModuleNotFoundError(
                "No module named 'pytorch_lightning' found on this machine. "
                "Run 'pip install pytorch_lightning' to install the library")

        if not has_val:
            self.val_dataloader = None

        if not has_test:
            self.test_dataloader = None

        if 'shuffle' in kwargs:
            warnings.warn(f"The 'shuffle={kwargs['shuffle']}' option is "
                          f"ignored in '{self.__class__.__name__}'. Remove it "
                          f"from the argument list to disable this warning")
            del kwargs['shuffle']

        if 'pin_memory' not in kwargs:
            kwargs['pin_memory'] = True

        if 'persistent_workers' not in kwargs:
            kwargs['persistent_workers'] = kwargs.get('num_workers', 0) > 0

        self.kwargs = kwargs

    def setup(self, stage: Optional[str] = None):
        from pytorch_lightning.plugins import DDPSpawnPlugin
        if not isinstance(self.trainer.training_type_plugin, DDPSpawnPlugin):
            raise NotImplementedError(
                f"'{self.__class__.__name__}' currently only supports the "
                f"'ddp_spawn' training strategy of 'pytorch_lightning'")

    def _kwargs_repr(self, **kwargs) -> str:
        kwargs = [f'{k}={v}' for k, v in kwargs.items() if v is not None]
        return ', '.join(kwargs)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self._kwargs_repr(**self.kwargs)})'


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
        val_dataset: (Dataset, optional) The validation dataset.
            (default: :obj:`None`)
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
    def __init__(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        batch_size: int = 1,
        num_workers: int = 0,
        **kwargs,
    ):
        super().__init__(
            has_val=val_dataset is not None,
            has_test=test_dataset is not None,
            batch_size=batch_size,
            num_workers=num_workers,
            **kwargs,
        )

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def dataloader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(dataset, shuffle=shuffle, **self.kwargs)

    def train_dataloader(self) -> DataLoader:
        from torch.utils.data import IterableDataset
        shuffle = not isinstance(self.train_dataset, IterableDataset)
        return self.dataloader(self.train_dataset, shuffle=shuffle)

    def val_dataloader(self) -> DataLoader:
        return self.dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self.dataloader(self.test_dataset, shuffle=False)

    def __repr__(self) -> str:
        kwargs_repr = self._kwargs_repr(train_dataset=self.train_dataset,
                                        val_dataset=self.val_dataset,
                                        test_dataset=self.test_dataset,
                                        **self.kwargs)
        return f'{self.__class__.__name__}({kwargs_repr})'
