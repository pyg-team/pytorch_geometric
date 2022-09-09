from typing import Any, Callable, Dict, Iterator, Optional, Tuple, Union

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
    r"""A data loader that performs neighborhood sampling from link information."""
    def __init__(
        self,
        data: Union[Data, HeteroData, Tuple[FeatureStore, GraphStore]],
        link_sampler: Optional[BaseSampler] = None,
        link_sampler_kwargs: Dict[str, Any] = None,
        initialize_sampler=True,
        edge_label_index: InputEdges = None,
        edge_label: OptTensor = None,
        edge_label_time: OptTensor = None,
        transform: Callable = None,
        filter_per_worker: bool = False,
        neg_sampling_ratio: float = 0.0,
        **kwargs,
    ):
        # Remove for PyTorch Lightning:
        if 'dataset' in kwargs:
            del kwargs['dataset']
        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']

        self.data = data

        # Initialize sampler with keyword arguments:
        # NOTE sampler is an attribute of 'DataLoader':
        self.link_sampler = link_sampler
        if initialize_sampler:
            self.link_sampler = self.link_sampler(self.data,
                                                  **link_sampler_kwargs)

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

        if ((edge_label_time is None) !=
            (link_sampler_kwargs.get('time_attr', None) is None)):
            raise ValueError("`edge_label_time` is specified but `time_attr` "
                             "is `None` or vice-versa. Both arguments need to "
                             "be specified for temporal sampling")

        super().__init__(
            Dataset(edge_label_index, edge_label, edge_label_time),
            collate_fn=self.collate_fn, **kwargs)

    def filter_fn(
        self,
        out: Union[SamplerOutput, HeteroSamplerOutput],
    ) -> Union[Data, HeteroData]:
        r"""Joins the sampled nodes with their corresponding features,
        returning the resulting (Data or HeteroData) object to be used
        downstream."""
        if isinstance(out, SamplerOutput):
            edge_label_index, edge_label = out.metadata
            data = filter_data(self.data, out.node, out.row, out.col, out.edge,
                               self.link_sampler.edge_permutation)
            data.edge_label_index = edge_label_index
            data.edge_label = edge_label

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
    def __init__(self, edge_label_index: torch.Tensor,
                 edge_label: torch.Tensor, edge_label_time: OptTensor = None):
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
