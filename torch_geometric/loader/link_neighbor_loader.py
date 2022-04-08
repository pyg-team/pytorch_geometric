from typing import Any, Callable, Iterator, List, Optional, Tuple, Union

import torch
from torch import Tensor

from torch_geometric.data import Data, HeteroData
from torch_geometric.loader.base import DataLoaderIterator
from torch_geometric.loader.neighbor_loader import NeighborSampler
from torch_geometric.loader.utils import filter_data, filter_hetero_data
from torch_geometric.typing import InputEdges, NumNeighbors, OptTensor


class LinkNeighborSampler(NeighborSampler):
    def __call__(self, query: List[Tuple[Tensor]]):
        query = [torch.tensor(s) for s in zip(*query)]
        if len(query) == 2:
            edge_label_index = torch.stack(query, dim=0)
            edge_label = None
        else:
            edge_label_index = torch.stack(query[:2], dim=0)
            edge_label = query[2]

        if issubclass(self.data_cls, Data):
            sample_fn = torch.ops.torch_sparse.neighbor_sample

            query_nodes = edge_label_index.view(-1)
            query_nodes, reverse = query_nodes.unique(return_inverse=True)

            node, row, col, edge = sample_fn(
                self.colptr,
                self.row,
                query_nodes,
                self.num_neighbors,
                self.replace,
                self.directed,
            )

            return node, row, col, edge, reverse.view(2, -1), edge_label

        elif issubclass(self.data_cls, HeteroData):
            sample_fn = torch.ops.torch_sparse.hetero_neighbor_sample
            node_dict, row_dict, col_dict, edge_dict = sample_fn(
                self.node_types,
                self.edge_types,
                self.colptr_dict,
                self.row_dict,
                {
                    self.input_type[0]: edge_label_index[0],
                    self.input_type[-1]: edge_label_index[1],
                },
                self.num_neighbors,
                self.num_hops,
                self.replace,
                self.directed,
            )
            return (node_dict, row_dict, col_dict, edge_dict, edge_label_index,
                    edge_label)


class LinkNeighborLoader(torch.utils.data.DataLoader):
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
        edge_label_index: InputEdges = None,
        edge_label: OptTensor = None,
        replace: bool = False,
        directed: bool = True,
        transform: Callable = None,
        neighbor_sampler: Optional[LinkNeighborSampler] = None,
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
        self.edge_label_index = edge_label_index
        self.edge_label = edge_label
        self.replace = replace
        self.directed = directed
        self.transform = transform
        self.neighbor_sampler = neighbor_sampler

        edge_type, edge_label_index = get_edge_label_index(
            data, edge_label_index)

        if neighbor_sampler is None:
            self.neighbor_sampler = LinkNeighborSampler(
                data, num_neighbors, replace, directed, edge_type,
                share_memory=kwargs.get('num_workers', 0) > 0)

        super().__init__(Dataset(edge_label_index, edge_label),
                         collate_fn=self.neighbor_sampler, **kwargs)

    def transform_fn(self, out: Any) -> Union[Data, HeteroData]:
        if isinstance(self.data, Data):
            node, row, col, edge, edge_label_index, edge_label = out
            data = filter_data(self.data, node, row, col, edge,
                               self.neighbor_sampler.perm)
            data.edge_label_index = edge_label_index
            if edge_label is not None:
                data.edge_label = edge_label

        elif isinstance(self.data, HeteroData):
            (node_dict, row_dict, col_dict, edge_dict, edge_label_index,
             edge_label) = out
            data = filter_hetero_data(self.data, node_dict, row_dict, col_dict,
                                      edge_dict,
                                      self.neighbor_sampler.perm_dict)
            edge_type = self.neighbor_sampler.input_type
            data[edge_type].edge_label_index = edge_label_index
            if edge_label is not None:
                data[edge_type].edge_label = edge_label

        return data if self.transform is None else self.transform(data)

    def _get_iterator(self) -> Iterator:
        return DataLoaderIterator(super()._get_iterator(), self.transform_fn)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


###############################################################################


class Dataset(torch.utils.data.Dataset):
    def __init__(self, edge_label_index: Tensor, edge_label: OptTensor = None):
        self.edge_label_index = edge_label_index
        self.edge_label = edge_label

    def __getitem__(self, idx: int) -> Tuple[int]:
        if self.edge_label is None:
            return self.edge_label_index[0, idx], self.edge_label_index[1, idx]
        else:
            return (self.edge_label_index[0, idx],
                    self.edge_label_index[1, idx], self.edge_label[idx])

    def __len__(self) -> int:
        return self.edge_label_index.size(1)


def get_edge_label_index(
    data: Union[Data, HeteroData],
    edge_label_index: InputEdges,
) -> Tuple[Optional[str], Tensor]:
    edge_type = None
    if isinstance(data, Data):
        if edge_label_index is None:
            return None, data.edge_index
        return None, edge_label_index

    assert edge_label_index is not None
    assert isinstance(edge_label_index, (list, tuple))

    if isinstance(edge_label_index[0], str):
        edge_type = edge_label_index
        return edge_type, data[edge_type].edge_index

    assert len(edge_label_index) == 2

    edge_type, edge_label_index = edge_label_index
    if edge_label_index is None:
        return edge_type, data[edge_type].edge_index

    return edge_type, edge_label_index
