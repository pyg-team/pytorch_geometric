from typing import Any, Callable, Iterator, List, Tuple, Union

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from torch_geometric.data import Data
from torch_geometric.loader.base import DataLoaderIterator
from torch_geometric.loader.neighbor_loader import NeighborSampler
from torch_geometric.loader.utils import filter_data
from torch_geometric.typing import InputEdges, NumNeighbors, OptTensor
from torch_geometric.utils import mask_to_index


class LinkNeighborSampler(NeighborSampler):
    def __init__(self, data: Data, input_edges: Tensor,
                 num_neighbors: NumNeighbors, replace: bool = False,
                 directed: bool = True, share_memory: bool = False):
        super().__init__(data, num_neighbors, replace, directed, share_memory)
        self.input_edges = input_edges

    def __call__(self, index: Union[List[int], Tensor]):

        # take start and end node from each edge then deduplicate
        node_index = torch.cat(
            [self.input_edges[index], self.input_edges[index]], dim=0)
        node_index = torch.unique(node_index)

        # get sampled graph
        node, row, col, edge, _ = super().__call__(node_index)
        return node, row, col, edge, len(index), index


class LinkNeighborLoader(DataLoader):
    def __init__(
        self,
        data: Data,
        num_neighbors: NumNeighbors,
        input_edges: InputEdges = None,
        input_edge_labels: OptTensor = None,
        replace: bool = False,
        directed: bool = True,
        transform: Callable = None,
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
        self.input_edges = self._get_input_edges(input_edges)
        self.input_edge_labels = input_edge_labels
        self.replace = replace
        self.directed = directed
        self.transform = transform
        self.neighbor_sampler = LinkNeighborSampler(
            data, self.input_edges, num_neighbors, replace, directed,
            share_memory=kwargs.get('num_workers', 0) > 0)
        index = range(self.input_edges.size()[1])

        super().__init__(index, collate_fn=self.neighbor_sampler, **kwargs)

    def _get_iterator(self) -> Iterator:
        return DataLoaderIterator(super()._get_iterator(), self.transform_fn)

    def transform_fn(self, out: Any) -> Tuple[Data, torch.Tensor]:
        node, row, col, edge, batch_size, index = out
        data = filter_data(self.data, node, row, col, edge,
                           self.neighbor_sampler.perm)

        data.batch_size = batch_size
        data = data if self.transform is None else self.transform(data)

        data.sampled_index = index

        edges = self.input_edges[:, index]
        data.sampled_edges = edges

        if self.input_edge_labels is not None:
            labels = self.input_edge_labels[index]
            data.sampled_edge_labels = labels

        return data

    def _get_input_edges(self, input_edges: InputEdges):

        if isinstance(self.data, Data):
            if input_edges is None:
                return self.data.edge_index

            input_size = input_edges.size()

            if len(input_size) == 1 and input_edges.dtype == torch.bool:
                return self.data.edge_index[mask_to_index(input_edges)]

            if len(input_edges.size()) == 2 and input_edges.size()[0] == 2:
                return input_edges

            raise ValueError("`input_edges` in unsupported format")

        raise NotImplementedError("self.data must be `Data` object"
                                  )  # TODO: Fix this before PR ready.
