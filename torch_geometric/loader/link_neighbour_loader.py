from typing import Callable, List, Any, Tuple

import torch
from torch import Tensor

from torch_geometric.utils.mask import mask_to_index
from torch_geometric.data import Data
from torch_geometric.loader.base import BaseDataLoader
from torch_geometric.loader.neighbor_loader import NeighborSampler
from torch_geometric.typing import InputEdges, NumNeighbors, OptTensor
from torch_geometric.loader.utils import filter_data


class LinkNeighborSampler(NeighborSampler):

    def __init__(
        self,
        data: Data,
        num_neighbors: NumNeighbors,
        replace: bool = False,
        directed: bool = True,
    ):
        super().__init__(data, num_neighbors, replace, directed)
        self.data = data
    
    def __call__(self, index: List[Tensor]):
        
        batch_size = len(index)

        # take start and end node from each edge then deduplicate
        row, col = self.data.edge_index
        node_index = torch.cat([row[index], col[index]], dim=0)
        node_index = torch.unique(node_index)

        # get sampled graph
        node, row, col, edge, _ = super().__call__(index)
        return node, row, col, edge, batch_size, index


class LinkNeighborLoader(BaseDataLoader):

    def __init__(
        self,
        data: Data,
        num_neighbors: NumNeighbors,
        input_edge_idx: InputEdges = None,
        input_edge_labels: OptTensor = None,
        replace: bool = False,
        directed: bool = True,
        transform: Callable = None,
        negative_sampling: bool = False,
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
        self.input_edge_idx = input_edge_idx
        self.input_edge_labels = input_edge_labels
        self.replace = replace
        self.directed = directed
        self.transform = transform
        self.neighbor_sampler = LinkNeighborSampler(data, num_neighbors,
                                                replace, directed)

        return super().__init__(
            self.get_input_edge_idx(), 
            collate_fn=self.neighbor_sampler, **kwargs)

    def transform_fn(self, out: Any) -> Tuple[Data, torch.Tensor]:
        node, row, col, edge, batch_size, index = out
        data = filter_data(self.data, node, row, col, edge,
                            self.neighbor_sampler.perm)

        data.batch_size = batch_size
        data = data if self.transform is None else self.transform(data)

        labels = self.get_input_edge_labels()[index]
        return (data, labels)

    def get_input_edge_idx(self):
        if self.input_edge_idx is None:
            return range(self.data.num_edges)
        if self.input_edge_idx.dtype == torch.bool:
            self.input_edge_idx = mask_to_index(self.input_edge_idx)
        return self.input_edge_idx

    def get_input_edge_labels(self):
        if self.input_edge_labels is None:
            return torch.Tensor([True] * self.data.num_edges)
        return self.input_edge_idx
