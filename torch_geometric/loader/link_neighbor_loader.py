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
    def __init__(self, data: Data, num_neighbors: NumNeighbors,
                 replace: bool = False, directed: bool = True,
                 force_order: bool = False):
        super().__init__(data, num_neighbors, replace, directed)
        self.force_order = force_order
        if self.force_order and self.perm is not None:
            inv_perm = torch.argsort(self.perm)
            inv_perm.to(self.device)
            self.inv_perm = inv_perm
        else:
            self.inv_perm = None

    def __call__(self, index: Union[List[int], Tensor]):
        # if force_order and permuted modify index to respect suffle
        if self.force_order and self.inv_perm is not None:
            index = self.inv_perm[index]

        # get edges
        if isinstance(index, list):
            index = torch.Tensor(index)
        col = torch.searchsorted(self.colptr, index)
        row = self.row[index]

        batch_size = len(index)

        # take start and end node from each edge then deduplicate
        node_index = torch.cat([row, col], dim=0)
        node_index = torch.unique(node_index)

        # get sampled graph
        node, row, col, edge, _ = super().__call__(node_index)
        return node, row, col, edge, batch_size, index


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
        force_order: bool = False,
        **kwargs,
    ):

        # Remove for PyTorch Lightning:
        if 'dataset' in kwargs:
            del kwargs['dataset']
        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']

        # Save for PyTorch Lightning:
        self.data = data
        self.num_neighbors = num_neighbors
        self.input_edges = self._get_input_edges(input_edges)
        self.input_edge_labels = self._get_edge_labels(input_edge_labels)
        self.replace = replace
        self.directed = directed
        self.transform = transform
        self.neighbor_sampler = LinkNeighborSampler(data, num_neighbors,
                                                    replace, directed,
                                                    force_order)

        super().__init__(self.input_edges, collate_fn=self.neighbor_sampler,
                         **kwargs)

    def _get_iterator(self) -> Iterator:
        return DataLoaderIterator(super()._get_iterator(), self.transform_fn)

    def transform_fn(self, out: Any) -> Tuple[Data, torch.Tensor]:
        node, row, col, edge, batch_size, index = out
        data = filter_data(self.data, node, row, col, edge,
                           self.neighbor_sampler.perm)

        data.batch_size = batch_size
        data = data if self.transform is None else self.transform(data)

        labels = self.input_edge_labels[index]
        edges = self.data.edge_index.T[torch.Tensor(self.input_edges).type(
            torch.long)[index]]

        data.edge_label_index = edges
        data.edge_labels = labels
        return data

    def _get_input_edges(self, input_edges):
        if input_edges is None:
            return range(self.data.num_edges)
        if input_edges.dtype == torch.bool:
            input_edges = mask_to_index(input_edges)
        return input_edges

    def _get_edge_labels(self, input_edge_labels):
        if input_edge_labels is None:
            return torch.Tensor([True] * self.data.num_edges)
        return input_edge_labels
