from typing import Any, Callable, Iterator, Optional, Tuple, Union

import torch
from torch import Tensor

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.feature_store import FeatureStore
from torch_geometric.data.graph_store import GraphStore
from torch_geometric.loader.base import DataLoaderIterator
from torch_geometric.loader.utils import (
    filter_custom_store,
    filter_data,
    filter_hetero_data,
)
from torch_geometric.sampler import NeighborSampler
from torch_geometric.sampler.base import (
    EdgeSamplerInput,
    HeteroSamplerOutput,
    SamplerOutput,
)
from torch_geometric.typing import InputEdges, NumNeighbors, OptTensor


class LinkNeighborLoader(torch.utils.data.DataLoader):
    r"""A link-based data loader derived as an extension of the node-based
    :class:`torch_geometric.loader.NeighborLoader`.
    This loader allows for mini-batch training of GNNs on large-scale graphs
    where full-batch training is not feasible.

    More specifically, this loader first selects a sample of edges from the
    set of input edges :obj:`edge_label_index` (which may or not be edges in
    the original graph) and then constructs a subgraph from all the nodes
    present in this list by sampling :obj:`num_neighbors` neighbors in each
    iteration.

    .. code-block:: python

        from torch_geometric.datasets import Planetoid
        from torch_geometric.loader import LinkNeighborLoader

        data = Planetoid(path, name='Cora')[0]

        loader = LinkNeighborLoader(
            data,
            # Sample 30 neighbors for each node for 2 iterations
            num_neighbors=[30] * 2,
            # Use a batch size of 128 for sampling training nodes
            batch_size=128,
            edge_label_index=data.edge_index,
        )

        sampled_data = next(iter(loader))
        print(sampled_data)
        >>> Data(x=[1368, 1433], edge_index=[2, 3103], y=[1368],
                 train_mask=[1368], val_mask=[1368], test_mask=[1368],
                 edge_label_index=[2, 128])

    It is additionally possible to provide edge labels for sampled edges, which
    are then added to the batch:

    .. code-block:: python

        loader = LinkNeighborLoader(
            data,
            num_neighbors=[30] * 2,
            batch_size=128,
            edge_label_index=data.edge_index,
            edge_label=torch.ones(data.edge_index.size(1))
        )

        sampled_data = next(iter(loader))
        print(sampled_data)
        >>> Data(x=[1368, 1433], edge_index=[2, 3103], y=[1368],
                 train_mask=[1368], val_mask=[1368], test_mask=[1368],
                 edge_label_index=[2, 128], edge_label=[128])

    The rest of the functionality mirrors that of
    :class:`~torch_geometric.loader.NeighborLoader`, including support for
    heterogenous graphs.

    .. note::
        :obj:`neg_sampling_ratio` is currently implemented in an approximate
        way, *i.e.* negative edges may contain false negatives.

    Args:
        data (torch_geometric.data.Data or torch_geometric.data.HeteroData):
            The :class:`~torch_geometric.data.Data` or
            :class:`~torch_geometric.data.HeteroData` graph object.
        num_neighbors (List[int] or Dict[Tuple[str, str, str], List[int]]): The
            number of neighbors to sample for each node in each iteration.
            In heterogeneous graphs, may also take in a dictionary denoting
            the amount of neighbors to sample for each individual edge type.
            If an entry is set to :obj:`-1`, all neighbors will be included.
        edge_label_index (Tensor or EdgeType or Tuple[EdgeType, Tensor]):
            The edge indices for which neighbors are sampled to create
            mini-batches.
            If set to :obj:`None`, all edges will be considered.
            In heterogeneous graphs, needs to be passed as a tuple that holds
            the edge type and corresponding edge indices.
            (default: :obj:`None`)
        edge_label (Tensor, optional): The labels of edge indices for
            which neighbors are sampled. Must be the same length as
            the :obj:`edge_label_index`. If set to :obj:`None` its set to
            `torch.zeros(...)` internally. (default: :obj:`None`)
        edge_label_time (Tensor, optional): The timestamps for edge indices
            for which neighbors are sampled. Must be the same length as
            :obj:`edge_label_index`. If set, temporal sampling will be
            used such that neighbors are guaranteed to fulfill temporal
            constraints, *i.e.*, neighbors have an earlier timestamp than
            the ouput edge. The :obj:`time_attr` needs to be set for this
            to work. (default: :obj:`None`)
        replace (bool, optional): If set to :obj:`True`, will sample with
            replacement. (default: :obj:`False`)
        directed (bool, optional): If set to :obj:`False`, will include all
            edges between all sampled nodes. (default: :obj:`True`)
        neg_sampling_ratio (float, optional): The ratio of sampled negative
            edges to the number of positive edges.
            If :obj:`neg_sampling_ratio > 0` and in case :obj:`edge_label`
            does not exist, it will be automatically created and represents a
            binary classification task (:obj:`1` = edge, :obj:`0` = no edge).
            If :obj:`neg_sampling_ratio > 0` and in case :obj:`edge_label`
            exists, it has to be a categorical label from :obj:`0` to
            :obj:`num_classes - 1`.
            After negative sampling, label :obj:`0` represents negative edges,
            and labels :obj:`1` to :obj:`num_classes` represent the labels of
            positive edges.
            Note that returned labels are of type :obj:`torch.float` for binary
            classification (to facilitate the ease-of-use of
            :meth:`F.binary_cross_entropy`) and of type
            :obj:`torch.long` for multi-class classification (to facilitate the
            ease-of-use of :meth:`F.cross_entropy`). (default: :obj:`0.0`).
        time_attr (str, optional): The name of the attribute that denotes
            timestamps for the nodes in the graph. Only used if
            :obj:`edge_label_time` is set. (default: :obj:`None`)
        transform (Callable, optional): A function/transform that takes in
            a sampled mini-batch and returns a transformed version.
            (default: :obj:`None`)
        is_sorted (bool, optional): If set to :obj:`True`, assumes that
            :obj:`edge_index` is sorted by column. This avoids internal
            re-sorting of the data and can improve runtime and memory
            efficiency. (default: :obj:`False`)
        filter_per_worker (bool, optional): If set to :obj:`True`, will filter
            the returning data in each worker's subprocess rather than in the
            main process.
            Setting this to :obj:`True` is generally not recommended:
            (1) it may result in too many open file handles,
            (2) it may slown down data loading,
            (3) it requires operating on CPU tensors.
            (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`, such as :obj:`batch_size`,
            :obj:`shuffle`, :obj:`drop_last` or :obj:`num_workers`.
    """
    def __init__(
        self,
        data: Union[Data, HeteroData, Tuple[FeatureStore, GraphStore]],
        num_neighbors: NumNeighbors,
        edge_label_index: InputEdges = None,
        edge_label: OptTensor = None,
        edge_label_time: OptTensor = None,
        replace: bool = False,
        directed: bool = True,
        neg_sampling_ratio: float = 0.0,
        time_attr: Optional[str] = None,
        transform: Callable = None,
        is_sorted: bool = False,
        filter_per_worker: bool = False,
        neighbor_sampler: Optional[NeighborSampler] = None,
        **kwargs,
    ):
        # Remove for PyTorch Lightning:
        if 'dataset' in kwargs:
            del kwargs['dataset']
        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']

        self.data = data

        # Save for PyTorch Lightning < 1.6:
        self.num_neighbors = num_neighbors
        self.edge_label = edge_label
        self.edge_label_index = edge_label_index
        self.edge_label_time = edge_label_time
        self.replace = replace
        self.directed = directed
        self.neg_sampling_ratio = neg_sampling_ratio
        self.transform = transform
        self.filter_per_worker = filter_per_worker
        self.neighbor_sampler = neighbor_sampler
        self.neg_sampling_ratio = neg_sampling_ratio

        # Get input edge type:
        edge_type, edge_label_index = get_edge_label_index(
            data, edge_label_index)
        if edge_label is None:
            edge_label = torch.zeros(edge_label_index.size(1),
                                     device=edge_label_index.device)
        self.input_type = edge_type

        if (edge_label_time is None) != (time_attr is None):
            raise ValueError("`edge_label_time` is specified but `time_attr` "
                             "is `None` or vice-versa. Both arguments need to "
                             "be specified for temporal sampling")

        if neighbor_sampler is None:
            self.neighbor_sampler = NeighborSampler(
                data,
                num_neighbors=num_neighbors,
                replace=replace,
                directed=directed,
                input_type=self.input_type,
                is_sorted=is_sorted,
                time_attr=time_attr,
                share_memory=kwargs.get('num_workers', 0) > 0,
            )

        super().__init__(
            Dataset(edge_label_index, edge_label, edge_label_time),
            collate_fn=self.collate_fn, **kwargs)

    def filter_fn(
        self,
        out: Union[SamplerOutput, HeteroSamplerOutput],
    ) -> Union[Data, HeteroData]:
        if isinstance(out, SamplerOutput):
            edge_label_index, edge_label = out.metadata
            data = filter_data(self.data, out.node, out.row, out.col, out.edge,
                               self.neighbor_sampler.edge_permutation)
            data.edge_label_index = edge_label_index
            data.edge_label = edge_label

        elif isinstance(out, HeteroSamplerOutput):
            edge_label_index, edge_label, edge_label_time = out.metadata
            if isinstance(self.data, HeteroData):
                data = filter_hetero_data(
                    self.data, out.node, out.row, out.col, out.edge,
                    self.neighbor_sampler.edge_permutation)
            else:  # Tuple[FeatureStore, GraphStore]
                data = filter_custom_store(*self.data, out.node, out.row,
                                           out.col, out.edge)

            edge_type = self.input_type
            data[edge_type].edge_label_index = edge_label_index
            data[edge_type].edge_label = edge_label
            if edge_label_time is not None:
                data[edge_type].edge_label_time = edge_label_time

        else:
            raise TypeError(f"'{self.__class__.__name__}'' found invalid "
                            f"type: '{type(out)}'")

        return data if self.transform is None else self.transform(data)

    def collate_fn(self, index: EdgeSamplerInput) -> Any:
        out = self.neighbor_sampler.sample_from_edges(
            index,
            negative_sampling_ratio=self.neg_sampling_ratio,
        )
        if self.filter_per_worker:
            # We execute `filter_fn` in the worker process.
            out = self.filter_fn(out)
        return out

    def _get_iterator(self) -> Iterator:
        if self.filter_per_worker:
            return super()._get_iterator()
        # We execute `filter_fn` in the main process.
        return DataLoaderIterator(super()._get_iterator(), self.filter_fn)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


###############################################################################


class Dataset(torch.utils.data.Dataset):
    def __init__(self, edge_label_index: Tensor, edge_label: Tensor,
                 edge_label_time: OptTensor = None):
        self.edge_label_index = edge_label_index
        self.edge_label = edge_label
        self.edge_label_time = edge_label_time

    def __getitem__(self, idx: int) -> Tuple[int]:
        if self.edge_label_time is None:
            return (
                self.edge_label_index[0, idx],
                self.edge_label_index[1, idx],
                self.edge_label[idx],
            )
        else:
            return (
                self.edge_label_index[0, idx],
                self.edge_label_index[1, idx],
                self.edge_label[idx],
                self.edge_label_time[idx],
            )

    def __len__(self) -> int:
        return self.edge_label_index.size(1)


def get_edge_label_index(
    data: Union[Data, HeteroData, Tuple[FeatureStore, GraphStore]],
    edge_label_index: InputEdges,
) -> Tuple[Optional[str], Tensor]:
    edge_type = None
    if isinstance(data, Data):
        if edge_label_index is None:
            return None, data.edge_index
        return None, edge_label_index

    assert edge_label_index is not None
    assert isinstance(edge_label_index, (list, tuple))

    if isinstance(data, HeteroData):
        if isinstance(edge_label_index[0], str):
            edge_type = edge_label_index
            edge_type = data._to_canonical(*edge_type)
            assert edge_type in data.edge_types
            return edge_type, data[edge_type].edge_index

        assert len(edge_label_index) == 2

        edge_type, edge_label_index = edge_label_index
        edge_type = data._to_canonical(*edge_type)

        if edge_label_index is None:
            return edge_type, data[edge_type].edge_index

        return edge_type, edge_label_index

    else:  # Tuple[FeatureStore, GraphStore]
        _, graph_store = data

        # Need the edge index in COO for LinkNeighborLoader:
        def _get_edge_index(edge_type):
            row_dict, col_dict, _ = graph_store.coo([edge_type])
            row = list(row_dict.values())[0]
            col = list(col_dict.values())[0]
            return torch.stack((row, col), dim=0)

        if isinstance(edge_label_index[0], str):
            edge_type = edge_label_index
            return edge_type, _get_edge_index(edge_type)

        assert len(edge_label_index) == 2
        edge_type, edge_label_index = edge_label_index

        if edge_label_index is None:
            return edge_type, _get_edge_index(edge_type)

        return edge_type, edge_label_index
