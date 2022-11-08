import copy
import inspect
import warnings
from typing import Optional, Tuple, Union

import torch
from torch import Tensor

from torch_geometric.data import Data, Dataset, HeteroData
from torch_geometric.data.feature_store import FeatureStore
from torch_geometric.data.graph_store import GraphStore
from torch_geometric.loader import RandomNodeSampler
from torch_geometric.loader import (
    LinkLoader,
    LinkNeighborLoader,
    NeighborLoader,
    NodeLoader,
)
from torch_geometric.loader.dataloader import DataLoader
from torch_geometric.loader.utils import get_edge_label_index, get_input_nodes
from torch_geometric.sampler import BaseSampler, NeighborSampler
from torch_geometric.typing import InputEdges, InputNodes

try:
    from pytorch_lightning import LightningDataModule as PLLightningDataModule
    no_pytorch_lightning = False
except (ImportError, ModuleNotFoundError):
    PLLightningDataModule = object
    no_pytorch_lightning = True


class LightningDataModule(PLLightningDataModule):
    def __init__(self, has_val: bool, has_test: bool, train_dataset: Dataset,
                 val_dataset: Optional[Dataset] = None,
                 test_dataset: Optional[Dataset] = None,
                 pred_dataset: Optional[Dataset] = None, **kwargs):
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

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.pred_dataset = pred_dataset
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

    def dataloader(self, dataset, **kwargs) -> DataLoader:
        """"""
        raise NotImplementedError(
            "This is an abstract method which is overridden by its child classes."
        )

    def train_dataloader(self) -> DataLoader:
        """"""
        shuffle = (self.kwargs.get('sampler', None) is None
                   and self.kwargs.get('batch_sampler', None) is None)

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
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            pred_dataset=pred_dataset,
            **kwargs,
        )

    def dataloader(self, dataset: Dataset, **kwargs) -> DataLoader:
        return DataLoader(dataset, **kwargs)

    def train_dataloader(self) -> DataLoader:
        """"""
        from torch.utils.data import IterableDataset
        shuffle = (not isinstance(self.train_dataset, IterableDataset)
                   and self.kwargs.get('sampler', None) is None
                   and self.kwargs.get('batch_sampler', None) is None)

        return self.dataloader(self.train_dataset, shuffle=shuffle,
                               **self.kwargs)

    def __repr__(self) -> str:
        kwargs = kwargs_repr(train_dataset=self.train_dataset,
                             val_dataset=self.val_dataset,
                             test_dataset=self.test_dataset, **self.kwargs)
        return f'{self.__class__.__name__}({kwargs})'


# TODO explicitly support Tuple[FeatureStore, GraphStore]
# TODO unify loaders and samplers
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
        input_val_nodes (torch.Tensor or str or (str, torch.Tensor)): The
            indices of validation nodes.
            If not given, will try to automatically infer them from the
            :obj:`data` object by searching for :obj:`val_mask`,
            :obj:`valid_mask`, :obj:`val_idx`, :obj:`valid_idx`,
            :obj:`val_index`, or :obj:`valid_index` attributes.
            (default: :obj:`None`)
        input_test_nodes (torch.Tensor or str or (str, torch.Tensor)): The
            indices of test nodes.
            If not given, will try to automatically infer them from the
            :obj:`data` object by searching for :obj:`test_mask`,
            :obj:`test_idx`, or :obj:`test_index` attributes.
            (default: :obj:`None`)
        input_pred_nodes (torch.Tensor or str or (str, torch.Tensor)): The
            indices of prediction nodes.
            If not given, will try to automatically infer them from the
            :obj:`data` object by searching for :obj:`pred_mask`,
            :obj:`pred_idx`, or :obj:`pred_index` attributes.
            (default: :obj:`None`)
        loader (str): The scalability technique to use (:obj:`"full"`,
            :obj:`"neighbor"`). (default: :obj:`"neighbor"`)
        custom_loader (DataLoader, optional): A custom loader object to
            generate mini-batches. If set, will ignore the :obj:`loader`
            option. (default: :obj:`None`)
        node_sampler (BaseSampler, optional): A custom sampler object to
            generate mini-batches. If set, will ignore the :obj:`loader`
            option. (default: :obj:`None`)
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
        input_pred_nodes: InputNodes = None,
        loader: str = "neighbor",
        custom_loader: Optional[DataLoader] = None,
        node_sampler: Optional[BaseSampler] = None,
        batch_size: int = 1,
        num_workers: int = 0,
        **kwargs,
    ):
        if node_sampler is not None and custom_loader is not None:
            raise ValueError(
                "Sampler and loader object given. Please choose either a loader or a sampler."
            )
        elif node_sampler is not None:
            loader = 'custom_sampler'
        elif custom_loader is not None:
            loader = "custom_loader"
            self.custom_loader = custom_loader

        assert loader in [
            'full', 'neighbor', 'custom_sampler', 'custom_loader'
        ]

        if input_train_nodes is None:
            input_train_nodes = infer_input(data, split='train')

        if input_val_nodes is None:
            input_val_nodes = infer_input(data, split='val')
            if input_val_nodes is None:
                input_val_nodes = infer_input(data, split='valid')

        if input_test_nodes is None:
            input_test_nodes = infer_input(data, split='test')

        if input_pred_nodes is None:
            input_pred_nodes = infer_input(data, split='pred')

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
            train_dataset=input_train_nodes,
            val_dataset=input_val_nodes,
            test_dataset=input_test_nodes,
            pred_dataset=input_pred_nodes,
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
            sampler_args = dict(inspect.signature(NeighborSampler).parameters)
            sampler_args.pop('data')
            sampler_args.pop('input_type')
            sampler_args.pop('share_memory')
            sampler_kwargs = {
                key: kwargs.get(key, param.default)
                for key, param in sampler_args.items()
            }

            self.neighbor_sampler = NeighborSampler(
                data=data,
                input_type=get_input_nodes(data, input_train_nodes)[0],
                share_memory=num_workers > 0,
                **sampler_kwargs,
            )
        elif node_sampler is not None:
            # TODO Consider renaming to `self.node_sampler`
            self.neighbor_sampler = node_sampler

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
        **kwargs,
    ) -> DataLoader:
        if self.loader == 'full':
            warnings.filterwarnings('ignore', '.*does not have many workers.*')
            warnings.filterwarnings('ignore', '.*data loading bottlenecks.*')

            kwargs.pop('sampler', None)
            kwargs.pop('batch_sampler', None)
            kwargs.pop('num_neighbors', None)

            return torch.utils.data.DataLoader(
                [self.data],
                collate_fn=lambda xs: xs[0],
                **kwargs,
            )

        elif self.loader == 'neighbor':
            return NeighborLoader(
                self.data,
                neighbor_sampler=self.neighbor_sampler,
                input_nodes=input_nodes,
                **kwargs,
            )

        elif self.loader == 'custom_sampler':
            return NodeLoader(
                self.data,
                node_sampler=self.neighbor_sampler,
                input_nodes=input_nodes,
                **kwargs,
            )

        elif self.loader == 'custom_loader':
            if issubclass(self.custom_loader, RandomNodeSampler):
                kwargs.pop("batch_size", None)
            return self.custom_loader(self.data, **kwargs)

        raise NotImplementedError

    def __repr__(self) -> str:
        kwargs = kwargs_repr(data=self.data, loader=self.loader, **self.kwargs)
        return f'{self.__class__.__name__}({kwargs})'


# TODO unify loaders and samplers
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
        input_train_labels (Tensor, optional):
            The labels of train edges. (default: :obj:`None`)
        input_train_time (Tensor, optional): The timestamp
            of train edges. (default: :obj:`None`)
        input_val_edges (Tensor or EdgeType or Tuple[EdgeType, Tensor]):
            The validation edges. (default: :obj:`None`)
        input_val_labels (Tensor, optional):
            The labels of validation edges. (default: :obj:`None`)
        input_val_time (Tensor, optional): The timestamp
            of validation edges. (default: :obj:`None`)
        input_test_edges (Tensor or EdgeType or Tuple[EdgeType, Tensor]):
            The test edges. (default: :obj:`None`)
        input_test_labels (Tensor, optional):
            The labels of test edges. (default: :obj:`None`)
        input_test_time (Tensor, optional): The timestamp
            of test edges. (default: :obj:`None`)
        input_pred_edges (Tensor or EdgeType or Tuple[EdgeType, Tensor]):
            The prediction edges. (default: :obj:`None`)
        input_pred_labels (Tensor, optional):
            The labels of prediction edges. (default: :obj:`None`)
        input_pred_time (Tensor, optional): The timestamp
            of prediction edges. (default: :obj:`None`)
        loader (str): The scalability technique to use (:obj:`"full"`,
            :obj:`"neighbor"`). (default: :obj:`"neighbor"`)
        custom_loader (DataLoader, optional): A custom loader object to
            generate mini-batches. If set, will ignore the :obj:`loader`
            option. (default: :obj:`None`)
        link_sampler (BaseSampler, optional): A custom sampler object to
            generate mini-batches. If set, will ignore the :obj:`loader`
            option. (default: :obj:`None`)
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        num_workers: How many subprocesses to use for data loading.
            :obj:`0` means that the data will be loaded in the main process.
            (default: :obj:`0`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.loader.LinkNeighborLoader`.
    """
    def __init__(
        self,
        data: Union[Data, HeteroData, Tuple[FeatureStore, GraphStore]],
        input_train_edges: InputEdges = None,
        input_train_labels: Tensor = None,
        input_train_time: Tensor = None,
        input_val_edges: InputEdges = None,
        input_val_labels: Tensor = None,
        input_val_time: Tensor = None,
        input_test_edges: InputEdges = None,
        input_test_labels: Tensor = None,
        input_test_time: Tensor = None,
        input_pred_edges: InputEdges = None,
        input_pred_labels: Tensor = None,
        input_pred_time: Tensor = None,
        loader: str = "neighbor",
        custom_loader: Optional[DataLoader] = None,
        link_sampler: Optional[BaseSampler] = None,
        batch_size: int = 1,
        num_workers: int = 0,
        **kwargs,
    ):
        if link_sampler is not None and custom_loader is not None:
            raise ValueError(
                "Sampler and loader object given. Please choose either a loader or a sampler."
            )
        elif link_sampler is not None:
            loader = 'custom_sampler'
        elif custom_loader is not None:
            loader = "custom_loader"
            self.custom_loader = custom_loader

        assert loader in [
            'full', 'neighbor', 'link_neighbor', 'custom_sampler',
            'custom_loader'
        ]

        if input_train_edges is None:
            input_train_edges = infer_input(data, split='train',
                                            input_type="edge_index")
        if input_train_labels is None:
            input_train_labels = infer_input(data, split="train",
                                             input_type="label")
        if input_train_time is None:
            input_train_time = infer_input(data, split="train",
                                           input_type="time")

        if input_val_edges is None:
            input_val_edges = infer_input(data, split='val',
                                          input_type="edge_index")
            if input_val_edges is None:
                input_val_edges = infer_input(data, split='valid',
                                              input_type="edge_index")
        if input_val_labels is None:
            input_val_labels = infer_input(data, split='val',
                                           input_type="label")
            if input_val_labels is None:
                input_val_labels = infer_input(data, split='valid',
                                               input_type="label")
        if input_val_time is None:
            input_val_time = infer_input(data, split='val', input_type="time")
            if input_val_time is None:
                input_val_time = infer_input(data, split='valid',
                                             input_type="time")

        if input_test_edges is None:
            input_test_edges = infer_input(data, split='test',
                                           input_type="edge_index")
        if input_test_labels is None:
            input_test_labels = infer_input(data, split="test",
                                            input_type="label")
        if input_test_time is None:
            input_test_time = infer_input(data, split="test",
                                          input_type="time")

        if input_pred_edges is None:
            input_pred_edges = infer_input(data, split='pred',
                                           input_type="edge_index")
        if input_pred_labels is None:
            input_pred_labels = infer_input(data, split="pred",
                                            input_type="label")
        if input_pred_time is None:
            input_pred_time = infer_input(data, split="pred",
                                          input_type="time")

        if input_train_edges is None:
            raise ValueError(f"No input edges found")

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
            has_val=input_val_edges is not None,
            has_test=input_test_edges is not None,
            batch_size=batch_size,
            num_workers=num_workers,
            train_dataset=(input_train_edges, input_train_labels,
                           input_train_time),
            val_dataset=(input_val_edges, input_val_labels,
                         input_val_time),
            test_dataset=(input_test_edges, input_test_labels,
                          input_test_time),
            pred_dataset=(input_pred_edges, input_pred_labels,
                          input_pred_time),
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

        if loader in ['neighbor', 'link_neighbor']:
            sampler_args = dict(inspect.signature(NeighborSampler).parameters)
            sampler_args.pop('data')
            sampler_args.pop('input_type')
            sampler_args.pop('share_memory')
            sampler_kwargs = {
                key: kwargs.get(key, param.default)
                for key, param in sampler_args.items()
            }
            self.neighbor_sampler = NeighborSampler(
                data=data,
                input_type=get_edge_label_index(data, input_train_edges)[0],
                share_memory=num_workers > 0,
                **sampler_kwargs,
            )
        elif link_sampler is not None:
            # TODO Consider renaming to `self.link_sampler`
            self.neighbor_sampler = link_sampler

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
        input_data: Tuple[InputEdges, Optional[Tensor], Optional[Tensor]],
        **kwargs,
    ) -> DataLoader:
        input_edges, input_labels, input_time = input_data

        neg_sampling_ratio = kwargs.pop('neg_sampling_ratio', 0.0)
        if input_labels is not None:
            neg_sampling_ratio = 0.0

        if self.loader == 'full':
            warnings.filterwarnings('ignore', '.*does not have many workers.*')
            warnings.filterwarnings('ignore', '.*data loading bottlenecks.*')

            kwargs.pop('sampler', None)
            kwargs.pop('batch_sampler', None)
            kwargs.pop('num_neighbors', None)

            data = self.data
            data.edge_label_index = input_edges
            data.edge_label = input_labels

            return torch.utils.data.DataLoader(
                [data],
                collate_fn=lambda xs: xs[0],
                **kwargs,
            )

        elif self.loader in ['neighbor', 'link_neighbor']:
            return LinkNeighborLoader(
                self.data,
                neighbor_sampler=self.neighbor_sampler,
                edge_label_index=input_edges,
                edge_label=input_labels,
                edge_label_time=input_time,
                neg_sampling_ratio=neg_sampling_ratio,
                **kwargs,
            )

        elif self.loader == 'custom_sampler':
            return LinkLoader(
                self.data,
                link_sampler=self.neighbor_sampler,
                edge_label_index=input_edges,
                edge_label=input_labels,
                edge_label_time=input_time,
                **kwargs,
            )
        elif self.loader == 'custom_loader':
            return self.custom_loader(self.data, **kwargs)

        raise NotImplementedError

    def __repr__(self) -> str:
        kwargs = kwargs_repr(data=self.data, loader=self.loader, **self.kwargs)
        return f'{self.__class__.__name__}({kwargs})'


###############################################################################


# TODO support Tuple[FeatureStore, GraphStore]
def infer_input(data: Union[Data, HeteroData], split: str,
                input_type: str = "node") -> Union[InputNodes, InputEdges]:
    if isinstance(data, tuple):
        return None

    attr_name: Optional[str] = None
    if input_type == "node":
        if f'{split}_mask' in data:
            attr_name = f'{split}_mask'
        elif f'{split}_idx' in data:
            attr_name = f'{split}_idx'
        elif f'{split}_index' in data:
            attr_name = f'{split}_index'
    else:
        for attr in data.keys:
            if input_type in attr and split in attr:
                attr_name = attr

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
