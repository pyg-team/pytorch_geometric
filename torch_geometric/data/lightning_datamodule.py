import warnings
from typing import Optional, Union

import torch

from torch_geometric.data import Data, Dataset, HeteroData
from torch_geometric.loader.dataloader import DataLoader
from torch_geometric.loader.neighbor_loader import (
    NeighborLoader, NeighborSampler, get_input_node_type
)
from torch_geometric.typing import InputNodes

try:
    from pytorch_lightning import LightningDataModule as PLLightningDataModule
    no_pytorch_lightning = False
except (ImportError, ModuleNotFoundError):
    PLLightningDataModule = object
    no_pytorch_lightning = True


class LightningDataModule(PLLightningDataModule):
    def __init__(self, has_val: bool, has_test: bool, **kwargs):
        super().__init__()

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

    def prepare_data(self):
        from pytorch_lightning.plugins import (
            DDPSpawnPlugin, SingleDevicePlugin
        )
        plugin = self.trainer.training_type_plugin
        if not isinstance(plugin, (SingleDevicePlugin, DDPSpawnPlugin)):
            raise NotImplementedError(
                f"'{self.__class__.__name__}' currently only supports "
                f"'{SingleDevicePlugin.__name__}' and "
                f"'{DDPSpawnPlugin.__name__}' training type plugins of "
                f"'pytorch_lightning' (got '{plugin.__class__.__name__}')")

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self._kwargs_repr(**self.kwargs)})'


class LightningDataset(LightningDataModule):
    r"""Converts a set of :class:`~torch_geometric.data.Dataset` objects into a
    :class:`pytorch_lightning.LightningDataModule` variant, which can be
    automatically used as a :obj:`datamodule` for multi-GPU graph-level
    training via `PyTorch Lightning <https://www.pytorchlightning.ai>`__.
    :class:`LightningDataset` will take care of providing mini-batches via
    :class:`~torch_geometric.loader.DataLoader`.

    .. note::

        Currently only the
        :class:`pytorch_lightning.plugins.SingleDevicePlugin` and
        :class:`pytorch_lightning.plugins.DDPSpawnPlugin` training type plugins
        of `PyTorch Lightning <https://pytorch-lightning.readthedocs.io/en/
        latest/guides/speed.html>`__ are supported in order to correctly
        share data across all devices/processes:

        .. code-block::

            import pytorch_lightning as pl
            trainer = pl.Trainer(strategy="ddp_spawn", gpus=4)
            trainer.fit(model, datamodule)

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
            :class:`torch_geometric.loader.DataLoader`.
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
        """"""
        from torch.utils.data import IterableDataset
        shuffle = not isinstance(self.train_dataset, IterableDataset)
        return self.dataloader(self.train_dataset, shuffle=shuffle)

    def val_dataloader(self) -> DataLoader:
        """"""
        return self.dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """"""
        return self.dataloader(self.test_dataset, shuffle=False)

    def __repr__(self) -> str:
        kwargs = kwargs_repr(train_dataset=self.train_dataset,
                             val_dataset=self.val_dataset,
                             test_dataset=self.test_dataset, **self.kwargs)
        return f'{self.__class__.__name__}({kwargs})'


class LightningNodeData(LightningDataModule):
    r"""Converts a :class:`~torch_geometric.data.Data` or
    :class:`~torch_geometric.data.HeteroData` object into a
    :class:`pytorch_lightning.LightningDataModule` variant, which can be
    automatically used as a :obj:`datamodule` for multi-GPU node-level
    training via `PyTorch Lightning <https://www.pytorchlightning.ai>`_.
    :class:`LightningDataset` will take care of providing mini-batches via
    :class:`~torch_geometric.loader.NeighborLoader`.

    .. note::

        Currently only the
        :class:`pytorch_lightning.plugins.SingleDevicePlugin` and
        :class:`pytorch_lightning.plugins.DDPSpawnPlugin` training type plugins
        of `PyTorch Lightning <https://pytorch-lightning.readthedocs.io/en/
        latest/guides/speed.html>`__ are supported in order to correctly
        share data across all devices/processes:

        .. code-block::

            import pytorch_lightning as pl
            trainer = pl.Trainer(strategy="ddp_spawn", gpus=4)
            trainer.fit(model, datamodule)

    Args:
        data (Data or HeteroData): The :class:`~torch_geometric.data.Data` or
            :class:`~torch_geometric.data.HeteroData` graph object.
        input_train_nodes (torch.Tensor or str or (str, torch.Tensor)): The
            indices of training nodes. If not given, will try to automatically
            infer them from the :obj:`data` object. (default: :obj:`None`)
        input_val_nodes (torch.Tensor or str or (str, torch.Tensor)): The
            indices of validation nodes. If not given, will try to
            automatically infer them from the :obj:`data` object.
            (default: :obj:`None`)
        input_test_nodes (torch.Tensor or str or (str, torch.Tensor)): The
            indices of test nodes. If not given, will try to automatically
            infer them from the :obj:`data` object. (default: :obj:`None`)
        loader (str): The scalability technique to use (:obj:`"full"`,
            :obj:`"neighbor"`). (default: :obj:`"neighbor"`)
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        num_workers: How many subprocesses to use for data loading.
            :obj:`0` means that the data will be loaded in the main process.
            (default: :obj:`0`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.loader.NeighborLoader`.
    """
    def __init__(
        self,
        data: Union[Data, HeteroData],
        input_train_nodes: InputNodes = None,
        input_val_nodes: InputNodes = None,
        input_test_nodes: InputNodes = None,
        loader: str = "neighbor",
        batch_size: int = 1,
        num_workers: int = 0,
        **kwargs,
    ):

        assert loader in ['full', 'neighbor']

        if input_train_nodes is None:
            input_train_nodes = infer_input_nodes(data, split='train')

        if input_val_nodes is None:
            input_val_nodes = infer_input_nodes(data, split='val')

        if input_val_nodes is None:
            input_val_nodes = infer_input_nodes(data, split='valid')

        if input_test_nodes is None:
            input_test_nodes = infer_input_nodes(data, split='test')

        if loader == 'full' and batch_size != 1:
            warnings.warn(f"Re-setting 'batch_size' to 1 in "
                          f"'{self.__class__.__name__}' for loader='full' "
                          f"(got '{batch_size}')")
            batch_size = 1

        if loader == 'full' and num_workers != 0:
            warnings.warn(f"Re-setting 'num_workers' to 0 in "
                          f"'{self.__class__.__name__}' for loader='full' "
                          f"(got '{num_workers}')")
            num_workers = 0

        super().__init__(
            has_val=input_val_nodes is not None,
            has_test=input_test_nodes is not None,
            batch_size=batch_size,
            num_workers=num_workers,
            **kwargs,
        )

        if loader == 'full':
            if kwargs.get('pin_memory', False):
                warnings.warn(f"Re-setting 'pin_memory' to 'False' in "
                              f"'{self.__class__.__name__}' for loader='full' "
                              f"(got 'True')")
            self.kwargs['pin_memory'] = False

        self.data = data
        self.loader = loader

        if loader == 'neighbor':
            self.neighbor_sampler = NeighborSampler(
                data=data,
                num_neighbors=kwargs.get('num_neighbors', None),
                replace=kwargs.get('replace', False),
                directed=kwargs.get('directed', True),
                input_node_type=get_input_node_type(input_train_nodes),
            )
        self.input_train_nodes = input_train_nodes
        self.input_val_nodes = input_val_nodes
        self.input_test_nodes = input_test_nodes

    def prepare_data(self):
        """"""
        if self.loader == 'full':
            if self.trainer.num_processes != 1 or self.trainer.num_gpus != 1:
                raise ValueError(f"'{self.__class__.__name__}' with loader="
                                 f"'full' requires training on a single GPU")
        super().prepare_data()

    def dataloader(self, input_nodes: InputNodes) -> DataLoader:
        if self.loader == 'full':
            warnings.filterwarnings('ignore', '.*does not have many workers.*')
            warnings.filterwarnings('ignore', '.*data loading bottlenecks.*')
            return torch.utils.data.DataLoader([self.data], shuffle=False,
                                               collate_fn=lambda xs: xs[0],
                                               **self.kwargs)

        if self.loader == 'neighbor':
            warnings.filterwarnings('ignore', '.*has `shuffle=True`.*')
            return NeighborLoader(self.data, input_nodes=input_nodes,
                                  neighbor_sampler=self.neighbor_sampler,
                                  shuffle=True, **self.kwargs)

        raise NotImplementedError

    def train_dataloader(self) -> DataLoader:
        """"""
        return self.dataloader(self.input_train_nodes)

    def val_dataloader(self) -> DataLoader:
        """"""
        return self.dataloader(self.input_val_nodes)

    def test_dataloader(self) -> DataLoader:
        """"""
        return self.dataloader(self.input_test_nodes)

    def __repr__(self) -> str:
        kwargs = kwargs_repr(data=self.data, loader=self.loader, **self.kwargs)
        return f'{self.__class__.__name__}({kwargs})'


###############################################################################


def infer_input_nodes(data: Union[Data, HeteroData], split: str) -> InputNodes:
    attr_name: Optional[str] = None
    if f'{split}_mask' in data:
        attr_name = f'{split}_mask'
    elif f'{split}_idx' in data:
        attr_name = f'{split}_idx'
    elif f'{split}_index' in data:
        attr_name = f'{split}_index'

    if attr_name is None:
        return None

    if isinstance(data, Data):
        return data[attr_name]
    if isinstance(data, HeteroData):
        input_nodes_dict = data.collect(attr_name)
        if len(input_nodes_dict) != 1:
            raise ValueError(f"Could not automatically determine the input "
                             f"nodes of {data} since there exists multiple "
                             f"types with attribute '{attr_name}'")
        return list(input_nodes_dict.items())[0]
    return None


def kwargs_repr(**kwargs) -> str:
    return ', '.join([f'{k}={v}' for k, v in kwargs.items() if v is not None])
