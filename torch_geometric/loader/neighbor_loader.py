from typing import Union, List, Dict, Tuple, Callable, Optional
from torch_geometric.typing import NodeType, EdgeType

import torch
from torch import Tensor

from torch_geometric.data import Data, HeteroData
from torch_geometric.loader.utils import edge_type_to_str
from torch_geometric.loader.utils import to_csc, to_hetero_csc
from torch_geometric.loader.utils import filter_data, filter_hetero_data


class NeighborLoader(torch.utils.data.DataLoader):
    r"""A data loader that performs neighbor sampling as introduced in the
    `"Inductive Representation Learning on Large Graphs"
    <https://arxiv.org/abs/1706.02216>`_ paper.
    This loader allows for mini-batch training of GNNs on large-scale graphs
    where full-batch training is not feasible.

    More specifically, :obj:`num_neighbors` denotes how much neighbors are
    sampled for each node in each iteration.
    :class:`~torch_geometric.loader.NeighborLoader` takes in this list of
    :obj:`num_neighbors` and iteratively samples :obj:`num_neighbors[i]` for
    each node involved in iteration :obj:`i - 1`.

    Sampled nodes are sorted based on the order in which they were sampled.
    In particular, the first :obj:`batch_size` nodes represent the set of
    original mini-batch nodes.

    .. code-block:: python

        from torch_geometric.datasets import Planetoid
        from torch_geometric.loader import NeighborLoader

        data = Planetoid(path, name='Cora')[0]

        loader = NeighborLoader(
            data,
            # Sample 30 neighbors for each node for 2 iterations
            num_neighbors=[30] * 2,
            # Use a batch size of 128 for sampling training nodes
            batch_size=128,
            input_nodes=data.train_mask,
        )

        sampled_data = next(iter(loader))
        print(sampled_data.batch_size)
        >>> 128

    By default, the data loader will only include the edges that were
    originally sampled (:obj:`directed = True`).
    This option should only be used in case the number of hops is equivalent to
    the number of GNN layers.
    In case the number of GNN layers is greater than the number of hops,
    consider setting :obj:`directed = False`, which will include all edges
    between all sampled nodes (but is slightly slower as a result).

    Furthermore, :class:`~torch_geometric.loader.NeighborLoader` works for both
    **homogeneous** graphs stored via :class:`~torch_geometric.data.Data` as
    well as **heterogeneous** graphs stored via
    :class:`~torch_geometric.data.HeteroData`.
    When operating in heterogeneous graphs, more fine-grained control over
    the amount of sampled neighbors of individual edge types is possible, but
    not necessary:

    .. code-block:: python

        from torch_geometric.datasets import OGB_MAG
        from torch_geometric.loader import NeighborLoader

        hetero_data = OGB_MAG(path)[0]

        loader = NeighborLoader(
            hetero_data,
            # Sample 30 neighbors for each node and edge type for 2 iterations
            num_neighbors={key: [30] * 2 for key in hetero_data.edge_types},
            # Use a batch size of 128 for sampling training nodes of type paper
            batch_size=128,
            input_nodes=('paper', hetero_data['paper'].train_mask),
        )

        sampled_hetero_data = next(iter(loader))
        print(sampled_hetero_data['paper'].batch_size)
        >>> 128

    .. note::

        For an example of using
        :class:`~torch_geometric.loader.NeighborLoader`, see
        `examples/hetero/to_hetero_mag.py <https://github.com/pyg-team/
        pytorch_geometric/blob/master/examples/hetero/to_hetero_mag.py>`_.

    Args:
        data (torch_geometric.data.Data or torch_geometric.data.HeteroData):
            The :class:`~torch_geometric.data.Data` or
            :class:`~torch_geometric.data.HeteroData` graph object.
        num_neighbors (List[int] or Dict[Tuple[str, str, str], List[int]]): The
            number of neighbors to sample for each node in each iteration.
            In heterogeneous graphs, may also take in a dictionary denoting
            the amount of neighbors to sample for each individual edge type.
            If an entry is set to :obj:`-1`, all neighbors will be included.
        input_nodes (torch.Tensor or str or Tuple[str, torch.Tensor]): The
            indices of nodes for which neighbors are sampled to create
            mini-batches.
            Needs to be either given as a :obj:`torch.LongTensor` or
            :obj:`torch.BoolTensor`.
            If set to :obj:`None`, all nodes will be considered.
            In heterogeneous graphs, needs to be passed as a tuple that holds
            the node type and node indices. (default: :obj:`None`)
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
        num_neighbors: Union[List[int], Dict[EdgeType, List[int]]],
        input_nodes: Union[Optional[Tensor], NodeType,
                           Tuple[NodeType, Optional[Tensor]]] = None,
        replace: bool = False,
        directed: bool = True,
        transform: Callable = None,
        **kwargs,
    ):
        if kwargs.get('num_workers', 0) > 0:
            torch.multiprocessing.set_sharing_strategy('file_system')

        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']
        if 'dataset' in kwargs:
            del kwargs['dataset']

        self.data = data
        self.num_neighbors = num_neighbors
        self.input_nodes = input_nodes
        self.replace = replace
        self.directed = directed
        self.transform = transform

        if isinstance(data, Data):
            self.sample_fn = torch.ops.torch_sparse.neighbor_sample
            # Convert the graph data into a suitable format for sampling.
            self.colptr, self.row, self.perm = to_csc(data)
            assert isinstance(num_neighbors, (list, tuple))
            assert input_nodes is None or isinstance(input_nodes, Tensor)
            if input_nodes is None:
                self.input_nodes = torch.arange(data.num_nodes)
            elif input_nodes.dtype == torch.bool:
                self.input_nodes = input_nodes.nonzero(as_tuple=False).view(-1)
            super().__init__(self.input_nodes.tolist(), collate_fn=self.sample,
                             **kwargs)

        else:  # `HeteroData`:
            self.node_types, self.edge_types = data.metadata()
            self.sample_fn = torch.ops.torch_sparse.hetero_neighbor_sample
            # Convert the graph data into a suitable format for sampling.
            # NOTE: Since C++ cannot take dictionaries with tuples as key as
            # input, edge type triplets are converted into single strings.
            out = to_hetero_csc(data)
            self.colptr_dict, self.row_dict, self.perm_dict = out
            if isinstance(num_neighbors, (list, tuple)):
                self.num_neighbors = {
                    key: num_neighbors
                    for key in self.edge_types
                }
            self.num_neighbors = {
                edge_type_to_str(key): value
                for key, value in self.num_neighbors.items()
            }
            self.num_hops = max([len(v) for v in self.num_neighbors.values()])
            if isinstance(input_nodes, str):
                self.input_nodes = (input_nodes, None)
            assert isinstance(self.input_nodes, (list, tuple))
            assert len(self.input_nodes) == 2
            assert isinstance(self.input_nodes[0], str)
            if self.input_nodes[1] is None:
                index = torch.arange(data[self.input_nodes[0]].num_nodes)
                self.input_nodes = (self.input_nodes[0], index)
            elif self.input_nodes[1].dtype == torch.bool:
                index = self.input_nodes[1].nonzero(as_tuple=False).view(-1)
                self.input_nodes = (self.input_nodes[0], index)
            super().__init__(self.input_nodes[1].tolist(),
                             collate_fn=self.hetero_sample, **kwargs)

    def sample(self, indices: List[int]) -> Data:
        node, row, col, edge = self.sample_fn(
            self.colptr,
            self.row,
            torch.tensor(indices),
            self.num_neighbors,
            self.replace,
            self.directed,
        )

        data = filter_data(self.data, node, row, col, edge, self.perm)
        data.batch_size = len(indices)
        data = data if self.transform is None else self.transform(data)

        return data

    def hetero_sample(self, indices: List[int]) -> HeteroData:
        node_dict, row_dict, col_dict, edge_dict = self.sample_fn(
            self.node_types,
            self.edge_types,
            self.colptr_dict,
            self.row_dict,
            {self.input_nodes[0]: torch.tensor(indices)},
            self.num_neighbors,
            self.num_hops,
            self.replace,
            self.directed,
        )

        data = filter_hetero_data(self.data, node_dict, row_dict, col_dict,
                                  edge_dict, self.perm_dict)
        data[self.input_nodes[0]].batch_size = len(indices)
        data = data if self.transform is None else self.transform(data)

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
