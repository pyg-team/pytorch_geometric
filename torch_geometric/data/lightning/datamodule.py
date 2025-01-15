import copy
import inspect
import warnings
from typing import Any, Dict, Optional, Tuple, Type, Union

import torch

from torch_geometric.data import Data, Dataset, HeteroData
from torch_geometric.loader import DataLoader, LinkLoader, NodeLoader
from torch_geometric.sampler import BaseSampler, NeighborSampler
from torch_geometric.typing import InputEdges, InputNodes, OptTensor

try:
    from pytorch_lightning import LightningDataModule as PLLightningDataModule
    no_pytorch_lightning = False
except ImportError:
    PLLightningDataModule = object  # type: ignore
    no_pytorch_lightning = True


class LightningDataModule(PLLightningDataModule):
    def __init__(self, has_val: bool, has_test: bool, **kwargs: Any) -> None:
        super().__init__()

        if no_pytorch_lightning:
            raise ModuleNotFoundError(
                "No module named 'pytorch_lightning' found on this machine. "
                "Run 'pip install pytorch_lightning' to install the library.")

        if not has_val:
            self.val_dataloader = None  # type: ignore

        if not has_test:
            self.test_dataloader = None  # type: ignore

        kwargs.setdefault('batch_size', 1)
        kwargs.setdefault('num_workers', 0)
        kwargs.setdefault('pin_memory', True)
        kwargs.setdefault('persistent_workers',
                          kwargs.get('num_workers', 0) > 0)

        if 'shuffle' in kwargs:
            warnings.warn(f"The 'shuffle={kwargs['shuffle']}' option is "
                          f"ignored in '{self.__class__.__name__}'. Remove it "
                          f"from the argument list to disable this warning")
            del kwargs['shuffle']

        self.kwargs = kwargs

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({kwargs_repr(**self.kwargs)})'


class LightningData(LightningDataModule):
    def __init__(
        self,
        data: Union[Data, HeteroData],
        has_val: bool,
        has_test: bool,
        loader: str = 'neighbor',
        graph_sampler: Optional[BaseSampler] = None,
        eval_loader_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault('batch_size', 1)
        kwargs.setdefault('num_workers', 0)

        if graph_sampler is not None:
            loader = 'custom'

        # For full-batch training, we use reasonable defaults for a lot of
        # data-loading options:
        if loader not in ['full', 'neighbor', 'link_neighbor', 'custom']:
            raise ValueError(f"Undefined 'loader' option (got '{loader}')")

        if loader == 'full' and kwargs['batch_size'] != 1:
            warnings.warn(f"Re-setting 'batch_size' to 1 in "
                          f"'{self.__class__.__name__}' for loader='full' "
                          f"(got '{kwargs['batch_size']}')")
            kwargs['batch_size'] = 1

        if loader == 'full' and kwargs['num_workers'] != 0:
            warnings.warn(f"Re-setting 'num_workers' to 0 in "
                          f"'{self.__class__.__name__}' for loader='full' "
                          f"(got '{kwargs['num_workers']}')")
            kwargs['num_workers'] = 0

        if loader == 'full' and kwargs.get('sampler') is not None:
            warnings.warn("'sampler' option is not supported for "
                          "loader='full'")
            kwargs.pop('sampler', None)

        if loader == 'full' and kwargs.get('batch_sampler') is not None:
            warnings.warn("'batch_sampler' option is not supported for "
                          "loader='full'")
            kwargs.pop('batch_sampler', None)

        super().__init__(has_val, has_test, **kwargs)

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

            # Define a new `NeighborSampler` to be re-used across data loaders:
            sampler_kwargs, self.loader_kwargs = split_kwargs(
                self.kwargs,
                NeighborSampler,
            )
            sampler_kwargs.setdefault('share_memory',
                                      self.kwargs['num_workers'] > 0)
            self.graph_sampler: BaseSampler = NeighborSampler(
                data, **sampler_kwargs)

        elif graph_sampler is not None:
            sampler_kwargs, self.loader_kwargs = split_kwargs(
                self.kwargs,
                graph_sampler.__class__,
            )
            if len(sampler_kwargs) > 0:
                warnings.warn(f"Ignoring the arguments "
                              f"{list(sampler_kwargs.keys())} in "
                              f"'{self.__class__.__name__}' since a custom "
                              f"'graph_sampler' was passed")
            self.graph_sampler = graph_sampler

        else:
            assert loader == 'full'
            self.loader_kwargs = self.kwargs

        # Determine validation sampler and loader arguments ###################

        self.eval_loader_kwargs = copy.copy(self.loader_kwargs)
        if eval_loader_kwargs is not None:
            # If the user wants to override certain values during evaluation,
            # we shallow-copy the graph sampler and update its attributes.
            if hasattr(self, 'graph_sampler'):
                self.eval_graph_sampler = copy.copy(self.graph_sampler)

                eval_sampler_kwargs, eval_loader_kwargs = split_kwargs(
                    eval_loader_kwargs,
                    self.graph_sampler.__class__,
                )
                for key, value in eval_sampler_kwargs.items():
                    setattr(self.eval_graph_sampler, key, value)

            self.eval_loader_kwargs.update(eval_loader_kwargs)

        elif hasattr(self, 'graph_sampler'):
            self.eval_graph_sampler = self.graph_sampler

        self.eval_loader_kwargs.pop('sampler', None)
        self.eval_loader_kwargs.pop('batch_sampler', None)

        if 'batch_sampler' in self.loader_kwargs:
            self.loader_kwargs.pop('batch_size', None)

    @property
    def train_shuffle(self) -> bool:
        shuffle = self.loader_kwargs.get('sampler', None) is None
        shuffle &= self.loader_kwargs.get('batch_sampler', None) is None
        return shuffle

    def prepare_data(self) -> None:
        if self.loader == 'full':
            assert self.trainer is not None
            try:
                num_devices = self.trainer.num_devices
            except AttributeError:
                # PyTorch Lightning < 1.6 backward compatibility:
                num_devices = self.trainer.num_processes  # type: ignore
                num_gpus = self.trainer.num_gpus  # type: ignore
                num_devices = max(num_devices, num_gpus)

            if num_devices > 1:
                raise ValueError(
                    f"'{self.__class__.__name__}' with loader='full' requires "
                    f"training on a single device")
        super().prepare_data()

    def full_dataloader(self, **kwargs: Any) -> torch.utils.data.DataLoader:
        warnings.filterwarnings('ignore', '.*does not have many workers.*')
        warnings.filterwarnings('ignore', '.*data loading bottlenecks.*')

        return torch.utils.data.DataLoader(
            [self.data],  # type: ignore
            collate_fn=lambda xs: xs[0],
            **kwargs,
        )

    def __repr__(self) -> str:
        kwargs = kwargs_repr(data=self.data, loader=self.loader, **self.kwargs)
        return f'{self.__class__.__name__}({kwargs})'


class LightningDataset(LightningDataModule):
    r"""Converts a set of :class:`~torch_geometric.data.Dataset` objects into a
    :class:`pytorch_lightning.LightningDataModule` variant. It can then be
    automatically used as a :obj:`datamodule` for multi-GPU graph-level
    training via :lightning:`null`
    `PyTorch Lightning <https://www.pytorchlightning.ai>`__.
    :class:`LightningDataset` will take care of providing mini-batches via
    :class:`~torch_geometric.loader.DataLoader`.

    .. note::

        Currently only the
        :class:`pytorch_lightning.strategies.SingleDeviceStrategy` and
        :class:`pytorch_lightning.strategies.DDPStrategy` training
        strategies of :lightning:`null` `PyTorch Lightning
        <https://pytorch-lightning.readthedocs.io/en/latest/guides/
        speed.html>`__ are supported in order to correctly share data across
        all devices/processes:

        .. code-block:: python

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
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.loader.DataLoader`.
    """
    def __init__(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        pred_dataset: Optional[Dataset] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            has_val=val_dataset is not None,
            has_test=test_dataset is not None,
            **kwargs,
        )

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.pred_dataset = pred_dataset

    def dataloader(self, dataset: Dataset, **kwargs: Any) -> DataLoader:
        return DataLoader(dataset, **kwargs)

    def train_dataloader(self) -> DataLoader:
        from torch.utils.data import IterableDataset

        shuffle = not isinstance(self.train_dataset, IterableDataset)
        shuffle &= self.kwargs.get('sampler', None) is None
        shuffle &= self.kwargs.get('batch_sampler', None) is None

        return self.dataloader(
            self.train_dataset,
            shuffle=shuffle,
            **self.kwargs,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None

        kwargs = copy.copy(self.kwargs)
        kwargs.pop('sampler', None)
        kwargs.pop('batch_sampler', None)

        return self.dataloader(self.val_dataset, shuffle=False, **kwargs)

    def test_dataloader(self) -> DataLoader:
        assert self.test_dataset is not None

        kwargs = copy.copy(self.kwargs)
        kwargs.pop('sampler', None)
        kwargs.pop('batch_sampler', None)

        return self.dataloader(self.test_dataset, shuffle=False, **kwargs)

    def predict_dataloader(self) -> DataLoader:
        assert self.pred_dataset is not None

        kwargs = copy.copy(self.kwargs)
        kwargs.pop('sampler', None)
        kwargs.pop('batch_sampler', None)

        return self.dataloader(self.pred_dataset, shuffle=False, **kwargs)

    def __repr__(self) -> str:
        kwargs = kwargs_repr(
            train_dataset=self.train_dataset,
            val_dataset=self.val_dataset,
            test_dataset=self.test_dataset,
            pred_dataset=self.pred_dataset,
            **self.kwargs,
        )
        return f'{self.__class__.__name__}({kwargs})'


class LightningNodeData(LightningData):
    r"""Converts a :class:`~torch_geometric.data.Data` or
    :class:`~torch_geometric.data.HeteroData` object into a
    :class:`pytorch_lightning.LightningDataModule` variant. It can then be
    automatically used as a :obj:`datamodule` for multi-GPU node-level
    training via :lightning:`null`
    `PyTorch Lightning <https://www.pytorchlightning.ai>`__.
    :class:`LightningDataset` will take care of providing mini-batches via
    :class:`~torch_geometric.loader.NeighborLoader`.

    .. note::

        Currently only the
        :class:`pytorch_lightning.strategies.SingleDeviceStrategy` and
        :class:`pytorch_lightning.strategies.DDPStrategy` training
        strategies of :lightning:`null` `PyTorch Lightning
        <https://pytorch-lightning.readthedocs.io/en/latest/guides/
        speed.html>`__ are supported in order to correctly share data across
        all devices/processes:

        .. code-block:: python

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
        loader: str = 'neighbor',
        node_sampler: Optional[BaseSampler] = None,
        eval_loader_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
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

        super().__init__(
            data=data,
            has_val=input_val_nodes is not None,
            has_test=input_test_nodes is not None,
            loader=loader,
            graph_sampler=node_sampler,
            eval_loader_kwargs=eval_loader_kwargs,
            **kwargs,
        )

        self.input_train_nodes = input_train_nodes
        self.input_train_time = input_train_time
        self.input_train_id: OptTensor = None

        self.input_val_nodes = input_val_nodes
        self.input_val_time = input_val_time
        self.input_val_id: OptTensor = None

        self.input_test_nodes = input_test_nodes
        self.input_test_time = input_test_time
        self.input_test_id: OptTensor = None

        self.input_pred_nodes = input_pred_nodes
        self.input_pred_time = input_pred_time
        self.input_pred_id: OptTensor = None

    def dataloader(
        self,
        input_nodes: InputNodes,
        input_time: OptTensor = None,
        input_id: OptTensor = None,
        node_sampler: Optional[BaseSampler] = None,
        **kwargs: Any,
    ) -> torch.utils.data.DataLoader:
        if self.loader == 'full':
            return self.full_dataloader(**kwargs)

        assert node_sampler is not None

        return NodeLoader(
            self.data,
            node_sampler=node_sampler,
            input_nodes=input_nodes,
            input_time=input_time,
            input_id=input_id,
            **kwargs,
        )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return self.dataloader(
            self.input_train_nodes,
            self.input_train_time,
            self.input_train_id,
            node_sampler=getattr(self, 'graph_sampler', None),
            shuffle=self.train_shuffle,
            **self.loader_kwargs,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return self.dataloader(
            self.input_val_nodes,
            self.input_val_time,
            self.input_val_id,
            node_sampler=getattr(self, 'eval_graph_sampler', None),
            shuffle=False,
            **self.eval_loader_kwargs,
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return self.dataloader(
            self.input_test_nodes,
            self.input_test_time,
            self.input_test_id,
            node_sampler=getattr(self, 'eval_graph_sampler', None),
            shuffle=False,
            **self.eval_loader_kwargs,
        )

    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        return self.dataloader(
            self.input_pred_nodes,
            self.input_pred_time,
            self.input_pred_id,
            node_sampler=getattr(self, 'eval_graph_sampler', None),
            shuffle=False,
            **self.eval_loader_kwargs,
        )


class LightningLinkData(LightningData):
    r"""Converts a :class:`~torch_geometric.data.Data` or
    :class:`~torch_geometric.data.HeteroData` object into a
    :class:`pytorch_lightning.LightningDataModule` variant. It can then be
    automatically used as a :obj:`datamodule` for multi-GPU link-level
    training via :lightning:`null`
    `PyTorch Lightning <https://www.pytorchlightning.ai>`__.
    :class:`LightningDataset` will take care of providing mini-batches via
    :class:`~torch_geometric.loader.LinkNeighborLoader`.

    .. note::

        Currently only the
        :class:`pytorch_lightning.strategies.SingleDeviceStrategy` and
        :class:`pytorch_lightning.strategies.DDPStrategy` training
        strategies of :lightning:`null` `PyTorch Lightning
        <https://pytorch-lightning.readthedocs.io/en/latest/guides/
        speed.html>`__ are supported in order to correctly share data across
        all devices/processes:

        .. code-block:: python

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
        eval_loader_kwargs (Dict[str, Any], optional): Custom keyword arguments
            that override the
            :class:`torch_geometric.loader.LinkNeighborLoader` configuration
            during evaluation. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.loader.LinkNeighborLoader`.
    """
    def __init__(
        self,
        data: Union[Data, HeteroData],
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
        loader: str = 'neighbor',
        link_sampler: Optional[BaseSampler] = None,
        eval_loader_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            data=data,
            has_val=input_val_edges is not None,
            has_test=input_test_edges is not None,
            loader=loader,
            graph_sampler=link_sampler,
            eval_loader_kwargs=eval_loader_kwargs,
            **kwargs,
        )

        self.input_train_edges = input_train_edges
        self.input_train_labels = input_train_labels
        self.input_train_time = input_train_time
        self.input_train_id: OptTensor = None

        self.input_val_edges = input_val_edges
        self.input_val_labels = input_val_labels
        self.input_val_time = input_val_time
        self.input_val_id: OptTensor = None

        self.input_test_edges = input_test_edges
        self.input_test_labels = input_test_labels
        self.input_test_time = input_test_time
        self.input_test_id: OptTensor = None

        self.input_pred_edges = input_pred_edges
        self.input_pred_labels = input_pred_labels
        self.input_pred_time = input_pred_time
        self.input_pred_id: OptTensor = None

    def dataloader(
        self,
        input_edges: InputEdges,
        input_labels: OptTensor = None,
        input_time: OptTensor = None,
        input_id: OptTensor = None,
        link_sampler: Optional[BaseSampler] = None,
        **kwargs: Any,
    ) -> torch.utils.data.DataLoader:
        if self.loader == 'full':
            return self.full_dataloader(**kwargs)

        assert link_sampler is not None

        return LinkLoader(
            self.data,
            link_sampler=link_sampler,
            edge_label_index=input_edges,
            edge_label=input_labels,
            edge_label_time=input_time,
            input_id=input_id,
            **kwargs,
        )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return self.dataloader(
            self.input_train_edges,
            self.input_train_labels,
            self.input_train_time,
            self.input_train_id,
            link_sampler=getattr(self, 'graph_sampler', None),
            shuffle=self.train_shuffle,
            **self.loader_kwargs,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return self.dataloader(
            self.input_val_edges,
            self.input_val_labels,
            self.input_val_time,
            self.input_val_id,
            link_sampler=getattr(self, 'eval_graph_sampler', None),
            shuffle=False,
            **self.eval_loader_kwargs,
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return self.dataloader(
            self.input_test_edges,
            self.input_test_labels,
            self.input_test_time,
            self.input_test_id,
            link_sampler=getattr(self, 'eval_graph_sampler', None),
            shuffle=False,
            **self.eval_loader_kwargs,
        )

    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        return self.dataloader(
            self.input_pred_edges,
            self.input_pred_labels,
            self.input_pred_time,
            self.input_pred_id,
            link_sampler=getattr(self, 'eval_graph_sampler', None),
            shuffle=False,
            **self.eval_loader_kwargs,
        )


###############################################################################


# TODO Support Tuple[FeatureStore, GraphStore]
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
        input_nodes_dict = {
            node_type: store[attr_name]
            for node_type, store in data.node_items() if attr_name in store
        }
        if len(input_nodes_dict) != 1:
            raise ValueError(f"Could not automatically determine the input "
                             f"nodes of {data} since there exists multiple "
                             f"types with attribute '{attr_name}'")
        return list(input_nodes_dict.items())[0]
    return None


def kwargs_repr(**kwargs: Any) -> str:
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
