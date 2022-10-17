from typing import Any, Callable, Iterator, Tuple, Union

import torch

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.feature_store import FeatureStore
from torch_geometric.data.graph_store import GraphStore
from torch_geometric.loader.base import DataLoaderIterator
from torch_geometric.loader.utils import (
    filter_custom_store,
    filter_data,
    filter_hetero_data,
    get_edge_label_index,
)
from torch_geometric.sampler.base import (
    BaseSampler,
    EdgeSamplerInput,
    HeteroSamplerOutput,
    SamplerOutput,
)
from torch_geometric.typing import InputEdges, OptTensor


class LinkLoader(torch.utils.data.DataLoader):
    r"""A data loader that performs neighbor sampling from link information,
    using a generic :class:`~torch_geometric.sampler.BaseSampler`
    implementation that defines a :meth:`sample_from_edges` function and is
    supported on the provided input :obj:`data` object.

    Args:
        data (torch_geometric.data.Data or torch_geometric.data.HeteroData):
            The :class:`~torch_geometric.data.Data` or
            :class:`~torch_geometric.data.HeteroData` graph object.
        link_sampler (torch_geometric.sampler.BaseSampler): The sampler
            implementation to be used with this loader. Note that the
            sampler implementation must be compatible with the input data
            object.
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
        neg_sampling_ratio (float, optional): the number of negative samples
            to include as a ratio of the number of positive examples
            (default: 0).
        transform (Callable, optional): A function/transform that takes in
            a sampled mini-batch and returns a transformed version.
            (default: :obj:`None`)
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
        link_sampler: BaseSampler,
        edge_label_index: InputEdges = None,
        edge_label: OptTensor = None,
        edge_label_time: OptTensor = None,
        neg_sampling_ratio: float = 0.0,
        transform: Callable = None,
        filter_per_worker: bool = False,
        **kwargs,
    ):
        # Remove for PyTorch Lightning:
        if 'dataset' in kwargs:
            del kwargs['dataset']
        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']

        self.data = data

        # Initialize sampler with keyword arguments:
        # NOTE sampler is an attribute of 'DataLoader', so we use link_sampler
        # here:
        self.link_sampler = link_sampler

        # Store additional arguments:
        self.edge_label = edge_label
        self.edge_label_index = edge_label_index
        self.edge_label_time = edge_label_time
        self.transform = transform
        self.filter_per_worker = filter_per_worker
        self.neg_sampling_ratio = neg_sampling_ratio

        # Get input type, or None for homogeneous graphs:
        edge_type, edge_label_index = get_edge_label_index(
            data, edge_label_index)
        if edge_label is None:
            edge_label = torch.zeros(edge_label_index.size(1),
                                     device=edge_label_index.device)
        self.input_type = edge_type

        super().__init__(
            Dataset(edge_label_index, edge_label, edge_label_time),
            collate_fn=self.collate_fn,
            **kwargs,
        )

    def filter_fn(
        self,
        out: Union[SamplerOutput, HeteroSamplerOutput],
    ) -> Union[Data, HeteroData]:
        r"""Joins the sampled nodes with their corresponding features,
        returning the resulting (Data or HeteroData) object to be used
        downstream."""
        if isinstance(out, SamplerOutput):
            edge_label_index, edge_label, edge_label_time = out.metadata
            data = filter_data(self.data, out.node, out.row, out.col, out.edge,
                               self.link_sampler.edge_permutation)
            data.batch = out.batch
            data.edge_label_index = edge_label_index
            data.edge_label = edge_label
            data.edge_label_time = edge_label_time

        elif isinstance(out, HeteroSamplerOutput):
            edge_label_index, edge_label, edge_label_time = out.metadata
            if isinstance(self.data, HeteroData):
                data = filter_hetero_data(self.data, out.node, out.row,
                                          out.col, out.edge,
                                          self.link_sampler.edge_permutation)
            else:  # Tuple[FeatureStore, GraphStore]
                data = filter_custom_store(*self.data, out.node, out.row,
                                           out.col, out.edge)

            edge_type = self.input_type
            for key, batch in (out.batch or {}).items():
                data[key].batch = batch
            data[edge_type].edge_label_index = edge_label_index
            data[edge_type].edge_label = edge_label
            if edge_label_time is not None:
                data[edge_type].edge_label_time = edge_label_time

        else:
            raise TypeError(f"'{self.__class__.__name__}'' found invalid "
                            f"type: '{type(out)}'")

        return data if self.transform is None else self.transform(data)

    def collate_fn(self, index: EdgeSamplerInput) -> Any:
        r"""Samples a subgraph from a batch of input nodes."""
        out = self.link_sampler.sample_from_edges(
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
    def __init__(
        self,
        edge_label_index: torch.Tensor,
        edge_label: torch.Tensor,
        edge_label_time: OptTensor = None,
    ):
        # NOTE see documentation of LinkLoader for details on these three
        # input parameters:
        self.edge_label_index = edge_label_index
        self.edge_label = edge_label
        self.edge_label_time = edge_label_time

    def __getitem__(
        self,
        idx: int,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
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
