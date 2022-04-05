from typing import (
    Any,
    Callable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from torch_geometric.data import Data, HeteroData
from torch_geometric.loader.base import DataLoaderIterator
from torch_geometric.loader.neighbor_loader import NeighborSampler
from torch_geometric.loader.utils import filter_data, filter_hetero_data
from torch_geometric.typing import (
    EdgeType,
    InputEdges,
    NumNeighbors,
    OptTensor,
)
from torch_geometric.utils import mask_to_index


class LinkNeighborSampler(NeighborSampler):
    def __init__(
        self,
        data: Union[Data, HeteroData],
        input_edges: Tensor,
        num_neighbors: NumNeighbors,
        replace: bool = False,
        directed: bool = True,
        input_edge_type=Optional[EdgeType],
        share_memory: bool = False,
    ):
        super().__init__(
            data,
            num_neighbors,
            replace,
            directed,
            "none",  # TODO: Needed because of assert in base class, remove?
            share_memory)
        self.input_edges = input_edges

        if issubclass(self.data_cls, HeteroData):
            assert_is_hetro_edge_type(input_edge_type)
            self.start_node_type = input_edge_type[0]
            self.end_node_type = input_edge_type[2]

    def __call__(self, index: Union[List[int], Tensor]):

        if issubclass(self.data_cls, Data):

            # take start and end node from each edge then deduplicate
            node_index = torch.cat(
                [self.input_edges[index], self.input_edges[index]], dim=0)
            node_index = torch.unique(node_index)

            # get sampled graph
            node, row, col, edge, _ = super().__call__(node_index)
            return node, row, col, edge, len(index), index

        elif issubclass(self.data_cls, HeteroData):

            start_nodes = self.input_edges[0][index]
            end_nodes = self.input_edges[1][index]
            print(self.start_node_type)
            print(self.end_node_type)

            sample_fn = torch.ops.torch_sparse.hetero_neighbor_sample
            node_dict, row_dict, col_dict, edge_dict = sample_fn(
                self.node_types,
                self.edge_types,
                self.colptr_dict,
                self.row_dict,
                {
                    self.start_node_type: start_nodes,
                    self.end_node_type: end_nodes
                },
                self.num_neighbors,
                self.num_hops,
                self.replace,
                self.directed,
            )
            return node_dict, row_dict, col_dict, edge_dict, len(index), index

        else:
            raise TypeError(
                f'NeighborLoader found invalid type: {self.data_cls}')


class LinkNeighborLoader(DataLoader):
    def __init__(
        self,
        data: Union[Data, HeteroData],
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
        edge_type, edges = self._get_input_edge_data(input_edges)
        self.input_edge_type = edge_type
        self.input_edges = edges
        self.input_edge_labels = input_edge_labels
        self.replace = replace
        self.directed = directed
        self.transform = transform
        self.neighbor_sampler = LinkNeighborSampler(
            data, self.input_edges, num_neighbors, replace, directed,
            self.input_edge_type,
            share_memory=kwargs.get('num_workers', 0) > 0)

        index = range(self.input_edges.size()[1])
        super().__init__(index, collate_fn=self.neighbor_sampler, **kwargs)

    def _get_iterator(self) -> Iterator:
        return DataLoaderIterator(super()._get_iterator(), self.transform_fn)

    def transform_fn(self, out: Any) -> Tuple[Data, torch.Tensor]:
        if isinstance(self.data, Data):

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

        elif isinstance(self.data, HeteroData):
            node_dict, row_dict, col_dict, edge_dict, batch_size, index = out

            print(index)

            data = filter_hetero_data(self.data, node_dict, row_dict, col_dict,
                                      edge_dict,
                                      self.neighbor_sampler.perm_dict)

            data[self.input_edge_type].batch_size = batch_size
            edges = self.input_edges[:, index]
            data[self.input_edge_type].sampled_edges = edges

            if self.input_edge_labels is not None:
                labels = self.input_edge_labels[index]
                data[self.input_edge_type].sampled_edge_labels = labels

        else:
            raise TypeError(
                f'LinkNeighborLoader found invalid type: {type(self.data)}')

        return data

    def _get_input_edge_data(self, input_edges: InputEdges):

        if isinstance(self.data, Data):
            if input_edges is None:
                return None, self.data.edge_index,
            input_size = input_edges.size()
            if len(input_size) == 1 and input_edges.dtype == torch.bool:
                return None, self.data.edge_index[mask_to_index(input_edges)]
            if len(input_edges.size()) == 2 and input_edges.size()[0] == 2:
                return None, input_edges

        elif isinstance(self.data, HeteroData):
            if isinstance(input_edges, (list, tuple)):
                if len(input_edges) == 3 and isinstance(input_edges[0], str):
                    return tuple(input_edges), self.data[tuple(
                        input_edges)].edge_index
            elif isinstance(input_edges, Sequence):
                return input_edges

        raise ValueError("`input_edges` in unsupported format")


def assert_is_hetro_edge_type(edge_type):
    assert (
        isinstance(edge_type, tuple) and len(edge_type) == 3
        and isinstance(edge_type[0], str) and isinstance(edge_type[1], str)
        and isinstance(edge_type[2], str)
    ), f"hetro data edge_type '{edge_type}' must be a tuple of 3 strings"
