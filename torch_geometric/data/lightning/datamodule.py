import copy
import inspect
import warnings
from typing import Any, Dict, Optional, Tuple, Type, Union

import torch

from torch_geometric.data import (
    Data,
    Dataset,
    FeatureStore,
    GraphStore,
    HeteroData,
)
from torch_geometric.loader import DataLoader, LinkLoader, NodeLoader
from torch_geometric.sampler import BaseSampler, NeighborSampler
from torch_geometric.typing import InputEdges, InputNodes, OptTensor

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
                "Run 'pip install pytorch_lightning' to install the library.")

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
        try:
            from pytorch_lightning.strategies import (
                DDPSpawnStrategy,
                SingleDeviceStrategy,
            )
            strategy = self.trainer.strategy
        except ImportError:
            # PyTorch Lightning < 1.6 backward compatibility:
            from pytorch_lightning.plugins import (
                DDPSpawnPlugin,
                SingleDevicePlugin,
            )
            DDPSpawnStrategy = DDPSpawnPlugin
            SingleDeviceStrategy = SingleDevicePlugin
            strategy = self.trainer.training_type_plugin

        if not isinstance(strategy, (SingleDeviceStrategy, DDPSpawnStrategy)):
            raise NotImplementedError(
                f"'{self.__class__.__name__}' currently only supports "
                f"'{SingleDeviceStrategy.__name__}' and "
                f"'{DDPSpawnStrategy.__name__}' training strategies of "
                f"'pytorch_lightning' (got '{strategy.__class__.__name__}')")

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
        :class:`pytorch_lightning.strategies.SingleDeviceStrategy` and
        :class:`pytorch_lightning.strategies.DDPSpawnStrategy` training
        strategies of `PyTorch Lightning
        <https://pytorch-lightning.readthedocs.io/en/latest/guides/
        speed.html>`__ are supported in order to correctly share data across
        all devices/processes:

        .. code-block::

            import pytorch_lightning as pl
            trainer = pl.Trainer(strategy="ddp_spawn", accelerator="gpu",
                                 devices=4)
            trainer.fit(model, datamodule)

    Args:
        train_dataset (Dataset): The training dataset.
        val_dataset (Dataset, optional): The validation dataset.
            (default: :obj:`None`)
        test_dataset (Dataset, optional): The test dataset.
            (default: :obj:`None`)
        pred_dataset (Dataset, optional): The prediction dataset.
            (default: :obj:`None`)
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        num_workers (int): How many subprocesses to use for data loading.
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
        pred_dataset: Optional[Dataset] = None,
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
        self.pred_dataset = pred_dataset

    def dataloader(self, dataset: Dataset, **kwargs) -> DataLoader:
        return DataLoader(dataset, **kwargs)

    def train_dataloader(self) -> DataLoader:
        """"""
        from torch.utils.data import IterableDataset

        shuffle = not isinstance(self.train_dataset, IterableDataset)
        shuffle &= self.kwargs.get('sampler', None) is None
        shuffle &= self.kwargs.get('batch_sampler', None) is None

        return self.dataloader(self.train_dataset, shuffle=shuffle,
                               **self.kwargs)

    def val_dataloader(self) -> DataLoader:
        """"""
        kwargs = copy.copy(self.kwargs)
        kwargs.pop('sampler', None)
        kwargs.pop('batch_sampler', None)

        return self.dataloader(self.val_dataset, shuffle=False, **kwargs)

    def test_dataloader(self) -> DataLoader:
        """"""
        kwargs = copy.copy(self.kwargs)
        kwargs.pop('sampler', None)
        kwargs.pop('batch_sampler', None)

        return self.dataloader(self.test_dataset, shuffle=False, **kwargs)

    def predict_dataloader(self) -> DataLoader:
        """"""
        kwargs = copy.copy(self.kwargs)
        kwargs.pop('sampler', None)
        kwargs.pop('batch_sampler', None)

        return self.dataloader(self.pred_dataset, shuffle=False, **kwargs)

    def __repr__(self) -> str:
        kwargs = kwargs_repr(train_dataset=self.train_dataset,
                             val_dataset=self.val_dataset,
                             test_dataset=self.test_dataset, **self.kwargs)
        return f'{self.__class__.__name__}({kwargs})'


# TODO explicitly support Tuple[FeatureStore, GraphStore]
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
        :class:`pytorch_lightning.strategies.SingleDeviceStrategy` and
        :class:`pytorch_lightning.strategies.DDPSpawnStrategy` training
        strategies of `PyTorch Lightning
        <https://pytorch-lightning.readthedocs.io/en/latest/guides/
        speed.html>`__ are supported in order to correctly share data across
        all devices/processes:

        .. code-block::

            import pytorch_lightning as pl
            trainer = pl.Trainer(strategy="ddp_spawn", accelerator="gpu",
                                 devices=4)
            trainer.fit(model, datamodule)

    Args:
        data (Data or HeteroData): The :class:`~torch_geometric.data.Data` or
            :class:`~torch_geometric.data.HeteroData` graph object.
        input_train_nodes (torch.Tensor or str or (str, torch.Tensor)): The
            indices of training nodes.
            If not given, will try to automatically infer them from the
            :obj:`data` object by searching for :obj:`train_mask`,
            :obj:`train_idx`, or :obj:`train_index` attributes.
            (default: :obj:`None`)
        input_train_time (torch.Tensor, optional): The timestamp
            of training nodes. (default: :obj:`None`)
        input_val_nodes (torch.Tensor or str or (str, torch.Tensor)): The
            indices of validation nodes.
            If not given, will try to automatically infer them from the
            :obj:`data` object by searching for :obj:`val_mask`,
            :obj:`valid_mask`, :obj:`val_idx`, :obj:`valid_idx`,
            :obj:`val_index`, or :obj:`valid_index` attributes.
            (default: :obj:`None`)
        input_val_time (torch.Tensor, optional): The timestamp
            of validation edges. (default: :obj:`None`)
        input_test_nodes (torch.Tensor or str or (str, torch.Tensor)): The
            indices of test nodes.
            If not given, will try to automatically infer them from the
            :obj:`data` object by searching for :obj:`test_mask`,
            :obj:`test_idx`, or :obj:`test_index` attributes.
            (default: :obj:`None`)
        input_test_time (torch.Tensor, optional): The timestamp
            of test nodes. (default: :obj:`None`)
        input_pred_nodes (torch.Tensor or str or (str, torch.Tensor)): The
            indices of prediction nodes.
            If not given, will try to automatically infer them from the
            :obj:`data` object by searching for :obj:`pred_mask`,
            :obj:`pred_idx`, or :obj:`pred_index` attributes.
            (default: :obj:`None`)
        input_pred_time (torch.Tensor, optional): The timestamp
            of prediction nodes. (default: :obj:`None`)
        loader (str): The scalability technique to use (:obj:`"full"`,
            :obj:`"neighbor"`). (default: :obj:`"neighbor"`)
        node_sampler (BaseSampler, optional): A custom sampler object to
            generate mini-batches. If set, will ignore the :obj:`loader`
            option. (default: :obj:`None`)
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        num_workers (int): How many subprocesses to use for data loading.
            :obj:`0` means that the data will be loaded in the main process.
            (default: :obj:`0`)
        eval_loader_kwargs (Dict[str, Any], optional): Custom keyword arguments
            that override the :class:`torch_geometric.loader.NeighborLoader`
            configuration during evaluation. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.loader.NeighborLoader`.
    """
    def __init__(
        self,
        data: Union[Data, HeteroData],
        input_train_nodes: InputNodes = None,
        input_train_time: OptTensor = None,
        input_val_nodes: InputNodes = None,
        input_val_time: OptTensor = None,
        input_test_nodes: InputNodes = None,
        input_test_time: OptTensor = None,
        input_pred_nodes: InputNodes = None,
        input_pred_time: OptTensor = None,
        loader: str = "neighbor",
        node_sampler: Optional[BaseSampler] = None,
        batch_size: int = 1,
        num_workers: int = 0,
        eval_loader_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        if node_sampler is not None:
            loader = 'custom'

        assert loader in ['full', 'neighbor', 'custom']

        if input_train_nodes is None:
            input_train_nodes = infer_input_nodes(data, split='train')

        if input_val_nodes is None:
            input_val_nodes = infer_input_nodes(data, split='val')
            if input_val_nodes is None:
                input_val_nodes = infer_input_nodes(data, split='valid')

        if input_test_nodes is None:
            input_test_nodes = infer_input_nodes(data, split='test')

        if input_pred_nodes is None:
            input_pred_nodes = infer_input_nodes(data, split='pred')

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

        if loader == 'full' and kwargs.get('sampler') is not None:
            warnings.warn("'sampler' option is not supported for "
                          "loader='full'")
            kwargs.pop('sampler', None)

        if loader == 'full' and kwargs.get('batch_sampler') is not None:
            warnings.warn("'batch_sampler' option is not supported for "
                          "loader='full'")
            kwargs.pop('sampler', None)

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

        # Determine sampler and loader arguments ##############################

        if loader == 'neighbor':
            # Define a new `NeighborSampler` that can be re-used across
            # different data loaders.
            sampler_kwargs, self.loader_kwargs = split_kwargs(
                self.kwargs,
                NeighborSampler,
            )
            sampler_kwargs.setdefault('share_memory', num_workers > 0)

            # TODO Consider renaming to `self.node_sampler`
            self.neighbor_sampler = NeighborSampler(data, **sampler_kwargs)

        elif node_sampler is not None:
            _, self.loader_kwargs = split_kwargs(
                self.kwargs,
                node_sampler.__class__,
            )
            self.neighbor_sampler = node_sampler

        else:
            self.loader_kwargs = self.kwargs

        # Determine validation sampler and loader arguments ###################

        self.eval_loader_kwargs = copy.copy(self.loader_kwargs)
        if eval_loader_kwargs is not None:
            # If the user wants to override certain values during evaluation,
            # we shallow-copy the sampler and update its attributes.
            if hasattr(self, 'neighbor_sampler'):
                self.eval_neighbor_sampler = copy.copy(self.neighbor_sampler)

                eval_sampler_kwargs, eval_loader_kwargs = split_kwargs(
                    eval_loader_kwargs,
                    self.neighbor_sampler.__class__,
                )
                for key, value in eval_sampler_kwargs.items():
                    setattr(self.eval_neighbor_sampler, key, value)

            self.eval_loader_kwargs.update(eval_loader_kwargs)

        elif hasattr(self, 'neighbor_sampler'):
            self.eval_neighbor_sampler = self.neighbor_sampler

        self.eval_loader_kwargs.pop('sampler', None)
        self.eval_loader_kwargs.pop('batch_sampler', None)

        #######################################################################

        self.input_train_nodes = input_train_nodes
        self.input_train_time = input_train_time
        self.input_val_nodes = input_val_nodes
        self.input_val_time = input_val_time
        self.input_test_nodes = input_test_nodes
        self.input_test_time = input_test_time
        self.input_pred_nodes = input_pred_nodes
        self.input_pred_time = input_pred_time

        # Can be overriden to set input indices of the `NodeLoader`:
        self.input_train_id: OptTensor = None
        self.input_val_id: OptTensor = None
        self.input_test_id: OptTensor = None
        self.input_pred_id: OptTensor = None

    def prepare_data(self):
        """"""
        if self.loader == 'full':
            try:
                num_devices = self.trainer.num_devices
            except AttributeError:
                # PyTorch Lightning < 1.6 backward compatibility:
                num_devices = self.trainer.num_processes
                num_devices = max(num_devices, self.trainer.num_gpus)

            if num_devices > 1:
                raise ValueError(
                    f"'{self.__class__.__name__}' with loader='full' requires "
                    f"training on a single device")
        super().prepare_data()

    def dataloader(
        self,
        input_nodes: InputNodes,
        input_time: OptTensor = None,
        input_id: OptTensor = None,
        node_sampler: Optional[BaseSampler] = None,
        **kwargs,
    ) -> DataLoader:
        if self.loader == 'full':
            warnings.filterwarnings('ignore', '.*does not have many workers.*')
            warnings.filterwarnings('ignore', '.*data loading bottlenecks.*')

            return torch.utils.data.DataLoader(
                [self.data],
                collate_fn=lambda xs: xs[0],
                **kwargs,
            )

        else:
            if node_sampler is None:
                warnings.warn("No 'node_sampler' specified. Falling back to "
                              "using the default training sampler.")
                node_sampler = self.neighbor_sampler

            return NodeLoader(
                self.data,
                node_sampler=node_sampler,
                input_nodes=input_nodes,
                input_time=input_time,
                input_id=input_id,
                **kwargs,
            )

    def train_dataloader(self) -> DataLoader:
        """"""
        shuffle = self.loader_kwargs.get('sampler', None) is None
        shuffle &= self.loader_kwargs.get('batch_sampler', None) is None

        return self.dataloader(
            self.input_train_nodes,
            self.input_train_time,
            self.input_train_id,
            node_sampler=getattr(self, 'neighbor_sampler', None),
            shuffle=shuffle,
            **self.loader_kwargs,
        )

    def val_dataloader(self) -> DataLoader:
        """"""
        return self.dataloader(
            self.input_val_nodes,
            self.input_val_time,
            self.input_val_id,
            node_sampler=getattr(self, 'eval_neighbor_sampler', None),
            shuffle=False,
            **self.eval_loader_kwargs,
        )

    def test_dataloader(self) -> DataLoader:
        """"""
        return self.dataloader(
            self.input_test_nodes,
            self.input_test_time,
            self.input_test_id,
            node_sampler=getattr(self, 'eval_neighbor_sampler', None),
            shuffle=False,
            **self.eval_loader_kwargs,
        )

    def predict_dataloader(self) -> DataLoader:
        """"""
        return self.dataloader(
            self.input_pred_nodes,
            self.input_pred_time,
            self.input_pred_id,
            node_sampler=getattr(self, 'eval_neighbor_sampler', None),
            shuffle=False,
            **self.eval_loader_kwargs,
        )

    def __repr__(self) -> str:
        kwargs = kwargs_repr(data=self.data, loader=self.loader, **self.kwargs)
        return f'{self.__class__.__name__}({kwargs})'


# TODO: Unify implementation with LightningNodeData via a common base class.
class LightningLinkData(LightningDataModule):
    r"""Converts a :class:`~torch_geometric.data.Data` or
    :class:`~torch_geometric.data.HeteroData` object into a
    :class:`pytorch_lightning.LightningDataModule` variant, which can be
    automatically used as a :obj:`datamodule` for multi-GPU link-level
    training (such as for link prediction) via `PyTorch Lightning
    <https://www.pytorchlightning.ai>`_. :class:`LightningDataset` will
    take care of providing mini-batches via
    :class:`~torch_geometric.loader.LinkNeighborLoader`.

    .. note::

        Currently only the
        :class:`pytorch_lightning.strategies.SingleDeviceStrategy` and
        :class:`pytorch_lightning.strategies.DDPSpawnStrategy` training
        strategies of `PyTorch Lightning
        <https://pytorch-lightning.readthedocs.io/en/latest/guides/
        speed.html>`__ are supported in order to correctly share data across
        all devices/processes:

        .. code-block::

            import pytorch_lightning as pl
            trainer = pl.Trainer(strategy="ddp_spawn", accelerator="gpu",
                                 devices=4)
            trainer.fit(model, datamodule)

    Args:
        data (Data or HeteroData or Tuple[FeatureStore, GraphStore]): The
            :class:`~torch_geometric.data.Data` or
            :class:`~torch_geometric.data.HeteroData` graph object, or a
            tuple of a :class:`~torch_geometric.data.FeatureStore` and
            :class:`~torch_geometric.data.GraphStore` objects.
        input_train_edges (Tensor or EdgeType or Tuple[EdgeType, Tensor]):
            The training edges. (default: :obj:`None`)
        input_train_labels (torch.Tensor, optional):
            The labels of training edges. (default: :obj:`None`)
        input_train_time (torch.Tensor, optional): The timestamp
            of training edges. (default: :obj:`None`)
        input_val_edges (Tensor or EdgeType or Tuple[EdgeType, Tensor]):
            The validation edges. (default: :obj:`None`)
        input_val_labels (torch.Tensor, optional):
            The labels of validation edges. (default: :obj:`None`)
        input_val_time (torch.Tensor, optional): The timestamp
            of validation edges. (default: :obj:`None`)
        input_test_edges (Tensor or EdgeType or Tuple[EdgeType, Tensor]):
            The test edges. (default: :obj:`None`)
        input_test_labels (torch.Tensor, optional):
            The labels of test edges. (default: :obj:`None`)
        input_test_time (torch.Tensor, optional): The timestamp
            of test edges. (default: :obj:`None`)
        input_pred_edges (Tensor or EdgeType or Tuple[EdgeType, Tensor]):
            The prediction edges. (default: :obj:`None`)
        input_pred_labels (torch.Tensor, optional):
            The labels of prediction edges. (default: :obj:`None`)
        input_pred_time (torch.Tensor, optional): The timestamp
            of prediction edges. (default: :obj:`None`)
        loader (str): The scalability technique to use (:obj:`"full"`,
            :obj:`"neighbor"`). (default: :obj:`"neighbor"`)
        link_sampler (BaseSampler, optional): A custom sampler object to
            generate mini-batches. If set, will ignore the :obj:`loader`
            option. (default: :obj:`None`)
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        num_workers (int): How many subprocesses to use for data loading.
            :obj:`0` means that the data will be loaded in the main process.
            (default: :obj:`0`)
        eval_loader_kwargs (Dict[str, Any], optional): Custom keyword arguments
            that override the
            :class:`torch_geometric.loader.LinkNeighborLoader` configuration
            during evaluation. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.loader.LinkNeighborLoader`.
    """
    def __init__(
        self,
        data: Union[Data, HeteroData, Tuple[FeatureStore, GraphStore]],
        input_train_edges: InputEdges = None,
        input_train_labels: OptTensor = None,
        input_train_time: OptTensor = None,
        input_val_edges: InputEdges = None,
        input_val_labels: OptTensor = None,
        input_val_time: OptTensor = None,
        input_test_edges: InputEdges = None,
        input_test_labels: OptTensor = None,
        input_test_time: OptTensor = None,
        input_pred_edges: InputEdges = None,
        input_pred_labels: OptTensor = None,
        input_pred_time: OptTensor = None,
        loader: str = "neighbor",
        link_sampler: Optional[BaseSampler] = None,
        batch_size: int = 1,
        num_workers: int = 0,
        eval_loader_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        if link_sampler is not None:
            loader = 'custom'

        assert loader in ['full', 'neighbor', 'link_neighbor', 'custom']

        if input_train_edges is None:
            raise NotImplementedError(f"'{self.__class__.__name__}' cannot "
                                      f"yet infer input edges automatically")

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

        if loader == 'full' and kwargs.get('sampler') is not None:
            warnings.warn("'sampler' option is not supported for "
                          "loader='full'")
            kwargs.pop('sampler', None)

        if loader == 'full' and kwargs.get('batch_sampler') is not None:
            warnings.warn("'batch_sampler' option is not supported for "
                          "loader='full'")
            kwargs.pop('sampler', None)

        super().__init__(
            has_val=input_val_edges is not None,
            has_test=input_test_edges is not None,
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

        # Determine sampler and loader arguments ##############################

        if loader in ['neighbor', 'link_neighbor']:
            # Define a new `NeighborSampler` that can be re-used across
            # different data loaders.
            sampler_kwargs, self.loader_kwargs = split_kwargs(
                self.kwargs,
                NeighborSampler,
            )
            sampler_kwargs.setdefault('share_memory', num_workers > 0)

            # TODO Consider renaming to `self.link_sampler`
            self.neighbor_sampler = NeighborSampler(data, **sampler_kwargs)

        elif link_sampler is not None:
            _, self.loader_kwargs = split_kwargs(
                self.kwargs,
                link_sampler.__class__,
            )
            self.neighbor_sampler = link_sampler

        else:
            self.loader_kwargs = self.kwargs

        # Determine validation sampler and loader arguments ###################

        self.eval_loader_kwargs = copy.copy(self.loader_kwargs)
        if eval_loader_kwargs is not None:
            # If the user wants to override certain values during evaluation,
            # we shallow-copy the sampler and update its attributes.
            if hasattr(self, 'neighbor_sampler'):
                self.eval_neighbor_sampler = copy.copy(self.neighbor_sampler)

                eval_sampler_kwargs, eval_loader_kwargs = split_kwargs(
                    eval_loader_kwargs,
                    self.neighbor_sampler.__class__,
                )
                for key, value in eval_sampler_kwargs.items():
                    setattr(self.eval_neighbor_sampler, key, value)

            self.eval_loader_kwargs.update(eval_loader_kwargs)

        elif hasattr(self, 'neighbor_sampler'):
            self.eval_neighbor_sampler = self.neighbor_sampler

        self.eval_loader_kwargs.pop('sampler', None)
        self.eval_loader_kwargs.pop('batch_sampler', None)

        self.input_train_edges = input_train_edges
        self.input_train_labels = input_train_labels
        self.input_train_time = input_train_time
        self.input_val_edges = input_val_edges
        self.input_val_labels = input_val_labels
        self.input_val_time = input_val_time
        self.input_test_edges = input_test_edges
        self.input_test_labels = input_test_labels
        self.input_test_time = input_test_time
        self.input_pred_edges = input_pred_edges
        self.input_pred_labels = input_pred_labels
        self.input_pred_time = input_pred_time

        # Can be overriden to set input indices of the `LinkLoader`:
        self.input_train_id: OptTensor = None
        self.input_val_id: OptTensor = None
        self.input_test_id: OptTensor = None
        self.input_pred_id: OptTensor = None

    def prepare_data(self):
        """"""
        if self.loader == 'full':
            try:
                num_devices = self.trainer.num_devices
            except AttributeError:
                # PyTorch Lightning < 1.6 backward compatibility:
                num_devices = self.trainer.num_processes
                num_devices = max(num_devices, self.trainer.num_gpus)

            if num_devices > 1:
                raise ValueError(
                    f"'{self.__class__.__name__}' with loader='full' requires "
                    f"training on a single device")
        super().prepare_data()

    def dataloader(
        self,
        input_edges: InputEdges,
        input_labels: OptTensor = None,
        input_time: OptTensor = None,
        input_id: OptTensor = None,
        link_sampler: Optional[BaseSampler] = None,
        **kwargs,
    ) -> DataLoader:
        if self.loader == 'full':
            warnings.filterwarnings('ignore', '.*does not have many workers.*')
            warnings.filterwarnings('ignore', '.*data loading bottlenecks.*')

            return torch.utils.data.DataLoader(
                [self.data],
                collate_fn=lambda xs: xs[0],
                **kwargs,
            )

        else:
            if link_sampler is None:
                warnings.warn("No 'link_sampler' specified. Falling back to "
                              "using the default training sampler.")
                link_sampler = self.neighbor_sampler

            return LinkLoader(
                self.data,
                link_sampler=link_sampler,
                edge_label_index=input_edges,
                edge_label=input_labels,
                edge_label_time=input_time,
                input_id=input_id,
                **kwargs,
            )

    def train_dataloader(self) -> DataLoader:
        """"""
        shuffle = self.loader_kwargs.get('sampler', None) is None
        shuffle &= self.loader_kwargs.get('batch_sampler', None) is None

        return self.dataloader(
            self.input_train_edges,
            self.input_train_labels,
            self.input_train_time,
            self.input_train_id,
            link_sampler=getattr(self, 'neighbor_sampler', None),
            shuffle=shuffle,
            **self.loader_kwargs,
        )

    def val_dataloader(self) -> DataLoader:
        """"""
        return self.dataloader(
            self.input_val_edges,
            self.input_val_labels,
            self.input_val_time,
            self.input_val_id,
            link_sampler=getattr(self, 'eval_neighbor_sampler', None),
            shuffle=False,
            **self.eval_loader_kwargs,
        )

    def test_dataloader(self) -> DataLoader:
        """"""
        return self.dataloader(
            self.input_test_edges,
            self.input_test_labels,
            self.input_test_time,
            self.input_test_id,
            link_sampler=getattr(self, 'eval_neighbor_sampler', None),
            shuffle=False,
            **self.eval_loader_kwargs,
        )

    def predict_dataloader(self) -> DataLoader:
        """"""
        return self.dataloader(
            self.input_pred_edges,
            self.input_pred_labels,
            self.input_pred_time,
            self.input_pred_id,
            link_sampler=getattr(self, 'eval_neighbor_sampler', None),
            shuffle=False,
            **self.eval_loader_kwargs,
        )

    def __repr__(self) -> str:
        kwargs = kwargs_repr(data=self.data, loader=self.loader, **self.kwargs)
        return f'{self.__class__.__name__}({kwargs})'


###############################################################################


# TODO support Tuple[FeatureStore, GraphStore]
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


def split_kwargs(
    kwargs: Dict[str, Any],
    sampler_cls: Type,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    r"""Splits keyword arguments into sampler and loader arguments."""
    sampler_args = inspect.signature(sampler_cls).parameters

    sampler_kwargs: Dict[str, Any] = {}
    loader_kwargs: Dict[str, Any] = {}

    for key, value in kwargs.items():
        if key in sampler_args:
            sampler_kwargs[key] = value
        else:
            loader_kwargs[key] = value

    return sampler_kwargs, loader_kwargs
