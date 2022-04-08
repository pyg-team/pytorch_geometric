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
            "link",  # prevents failing assert
            share_memory)
        self.input_edges = input_edges

        if issubclass(self.data_cls, HeteroData):
            assert_is_hetro_edge_type(input_edge_type)
            self.input_edge_type = input_edge_type
            self.start_node_type = input_edge_type[0]
            self.end_node_type = input_edge_type[2]

    def __call__(self, index: Union[List[int], Tensor]):

        if issubclass(self.data_cls, Data):
            # take start and end node from each edge then deduplicate
            query_nodes = torch.cat(
                [self.input_edges[0][index], self.input_edges[1][index]],
                dim=0)
            query_nodes, reverse_query_nodes = torch.unique(
                query_nodes, return_inverse=True)

            # get sampled graph
            node, row, col, edge, _ = super().__call__(query_nodes)

            # assume order of nodes in the new graph is same as input
            out_graph_nodes = torch.argsort(query_nodes)
            out_edges = out_graph_nodes[reverse_query_nodes].reshape(2, -1)

            return node, row, col, edge, len(index), out_edges, index

        elif issubclass(self.data_cls, HeteroData):

            query_start_nodes = self.input_edges[0][index]
            query_end_nodes = self.input_edges[1][index]

            sample_fn = torch.ops.torch_sparse.hetero_neighbor_sample
            node_dict, row_dict, col_dict, edge_dict = sample_fn(
                self.node_types,
                self.edge_types,
                self.colptr_dict,
                self.row_dict,
                {
                    self.start_node_type: query_start_nodes,
                    self.end_node_type: query_end_nodes
                },
                self.num_neighbors,
                self.num_hops,
                self.replace,
                self.directed,
            )

            out_edges = torch.stack([
                torch.argsort(
                    node_dict[self.start_node_type][:len(query_start_nodes)]),
                torch.argsort(
                    node_dict[self.end_node_type][:len(query_end_nodes)])
            ])
            return node_dict, row_dict, col_dict, edge_dict, len(
                index), out_edges, index

        else:
            raise TypeError(
                f'NeighborLoader found invalid type: {self.data_cls}')


class LinkNeighborLoader(DataLoader):
    r"""A link based data loader that is an extension of the node based
    :obj:`NeighborLoader`. This loader allows for mini-batch training of GNNs
    on large-scale graphs with respect to edge based tasks like link
    prediction.

    This loader first selects a sample of edges from the input list (which
    may or not be edges in the original graph) and then constructs a subgraph
    from all the nodes represented by this list by using
    :obj:`num_neighbors` neighbours in each iteration.

    .. code-block:: python

        from torch_geometric.datasets import Planetoid
        from torch_geometric.loader import NeighborLoader

        data = Planetoid(path, name='Cora')[0]

        loader = LinkNeighborLoader(
            data,
            # Sample 30 neighbors for each node for 2 iterations
            num_neighbors=[30] * 2,
            # Use a batch size of 128 for sampling training nodes
            batch_size=128,
            input_edges=data.edge_index,
        )

        sampled_data = next(iter(loader))
        print(sampled_data)
        >>> Data(x=[1368, 1433], edge_index=[2, 3103], y=[1368],
            train_mask=[1368], val_mask=[1368], test_mask=[1368],
            batch_size=128, sampled_edges=[2, 128])

    The batch size above is the number of edges that were sampled,
    while the sampled_edges gives the edge index of the edges that
    that subgraph has been built for.

    It is additionally possible to provide edge labels, which are
    then added to the batch.

    .. code-block:: python

        loader = LinkNeighborLoader(
            data,
            num_neighbors=[30] * 2,
            batch_size=128,
            input_edges=data.edge_index,
            input_edge_labels=torch.ones(data.edge_index.size()[1])
        )

        sampled_data = next(iter(loader))
        print(sampled_data)
        >>> Data(x=[1368, 1433], edge_index=[2, 3103], y=[1368],
            train_mask=[1368], val_mask=[1368], test_mask=[1368],
            batch_size=128, sampled_edges=[2, 128], sampled_edge_labels=[128])

    The rest of the functionality is mirros that of :obj:`NeighborLoader`,
    including support for hetrogenous graphs.

    Args:
        data (torch_geometric.data.Data or torch_geometric.data.HeteroData):
            The :class:`~torch_geometric.data.Data` or
            :class:`~torch_geometric.data.HeteroData` graph object.
        num_neighbors (List[int] or Dict[Tuple[str, str, str], List[int]]): The
            number of neighbors to sample for each node in each iteration.
            In heterogeneous graphs, may also take in a dictionary denoting
            the amount of neighbors to sample for each individual edge type.
            If an entry is set to :obj:`-1`, all neighbors will be included.
        input_edges (torch.Tensor or str or Tuple[Tuple(str), torch.Tensor]):
            The edge_index format edges for which neighbors are sampled to
            create mini-batches.
            If set to :obj:`None`, all edges will be considered.
            In heterogeneous graphs, needs to be passed as a tuple that holds
            the edge type and edge indices. (default: :obj:`None`)
            Note that the edges provide do not need to be in the graph provided
            to sample, but the nodes of those edge should be.
        input_edge_labels (torch.Tensor):
            The labels of the input_edges. Must be the same length as the
            input_edges.
            If set to :obj:`None` then no labels are returned in the batch.
        replace (bool, optional): If set to :obj:`True`, will sample with
            replacement. (default: :obj:`False`)
        directed (bool, optional): If set to :obj:`False`, will include all
            edges between all sampled nodes. (default: :obj:`True`)
        transform (Callable, optional): A function/transform that takes in
            a sampled mini-batch and returns a transformed version.
            (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`, such as :obj:`batch_size`,
            :obj:`shuffle`, :obj:`drop_last` or :obj:`num_workers`.
    """
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

            node, row, col, edge, batch_size, out_edges, index = out
            data = filter_data(self.data, node, row, col, edge,
                               self.neighbor_sampler.perm)

            data.batch_size = batch_size
            data = data if self.transform is None else self.transform(data)

            data.sampled_edges = out_edges
            if self.input_edge_labels is not None:
                labels = self.input_edge_labels[index]
                data.sampled_edge_labels = labels

        elif isinstance(self.data, HeteroData):
            node_d, row_d, col_d, edge_d, batch_size, out_edges, index = out

            data = filter_hetero_data(self.data, node_d, row_d, col_d, edge_d,
                                      self.neighbor_sampler.perm_dict)

            data[self.input_edge_type].batch_size = batch_size
            data[self.input_edge_type].sampled_edges = out_edges

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

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


def assert_is_hetro_edge_type(edge_type):
    assert (
        isinstance(edge_type, tuple) and len(edge_type) == 3
        and isinstance(edge_type[0], str) and isinstance(edge_type[1], str)
        and isinstance(edge_type[2], str)
    ), f"hetro data edge_type '{edge_type}' must be a tuple of 3 strings"
