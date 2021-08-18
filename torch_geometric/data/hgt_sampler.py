from typing import Union, Dict, List, Tuple, Callable, Optional

import copy

import torch
from torch import Tensor
from torch_sparse import SparseTensor
from torch_geometric.data import HeteroData
from torch_geometric.typing import NodeType


class HGTSampler(torch.utils.data.DataLoader):
    r"""The Heterogeneous Graph Sampler from the `"Heterogeneous Graph
    Transformer" <https://arxiv.org/abs/2003.01332>`_ paper, which allows for
    mini-batch training of heterogeneous GNNs on large-scale graphs where
    full-batch training is not feasible.

    :class:`~torch_geometric.data.HGTSampler` tries to (1) keep a similar
    number of nodes and edges for each type and (2) keep the sampled sub-graph
    dense to minimize the information loss and reduce the sample variance.

    Methodically, :class:`~torch_geometric.data.HGTSampler` keeps track of a
    node budget for each node type, which is then used to determine the
    sampling probability of a node.
    In particular, the probability of sampling a node is determined by the
    number of connections to already sampled nodes and their node degrees.
    With this, :class:`~torch_geometric.data.HGTSampler` will sample a fixed
    amount of neighbors for each node type in each iteration, as given by the
    :obj:`num_samples` argument.

    .. note::

        For an example of using :class:`~torch_geometric.data.HGTSampler`, see
        `examples/hetero/to_hetero_mag.py
        <https://github.com/rusty1s/pytorch_geometric/blob/master/examples/
        hetero/to_hetero_mag.py>`_.

    .. code-block:: python

        from torch_geometric.data import HGTSampler
        from torch_geometric.datasets import OGB_MAG

        hetero_data = OGB_MAG(path)[0]

        loader = HGTSampler(
            hetero_data,
            # Sample 512 nodes per type and per iteration for 4 iterations
            num_samples={key: [512] * 4 for key in hetero_data.node_types},
            # Use a batch size of 128 for sampling training nodes of type paper
            batch_size=128,
            input_nodes: ('paper': hetero_data['paper'].train_mask),
        )

        sampled_hetero_data = next(iter(loader))

    Args:
        data (torch_geometric.data.HeteroData): The
            :class:`~torch_geometric.data.HeteroData` graph data object.
        num_samples (List[int] or Dict[str, List[int]]): The number of nodes to
            sample in each iteration and for each node type.
            If given as a list, will sample the same amount of nodes for each
            node type.
        input_nodes (Tuple[str, torch.Tensor]): The nodes of a specific node
            type that should be considered for creating mini-batches.
            Needs to be either given as a :obj:`torch.LongTensor` or
            :obj:`torch.BoolTensor`.
            If node indices are set to :obj:`None`, all nodes will be
            considered.
        input_nodes (torch.Tensor or str or Tuple[str, torch.Tensor]): The
            indices of nodes for which neighbors are sampled to create
            mini-batches.
            Needs to be passed as a tuple that holds the node type and
            corresponding node indices.
            Node indices need to be either given as a :obj:`torch.LongTensor`
            or :obj:`torch.BoolTensor`.
        transform (Callable, optional): A function/transform that takes in
            an a sampled mini-batch and returns a transformed version.
            (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`, such as :obj:`batch_size`,
            :obj:`shuffle`, :obj:`drop_last` or :obj:`num_workers`.
    """
    def __init__(
        self,
        data: HeteroData,
        num_samples: Union[List[int], Dict[NodeType, List[int]]],
        input_nodes: Tuple[NodeType, Optional[Tensor]],
        transform: Callable = None,
        **kwargs,
    ):
        if kwargs.get('num_workers', 0) > 0:
            torch.multiprocessing.set_sharing_strategy('file_system')

        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']

        if isinstance(num_samples, (list, tuple)):
            num_samples = {key: num_samples for key in data.node_types}

        assert isinstance(input_nodes, (list, tuple))
        assert len(input_nodes) == 2
        assert isinstance(input_nodes[0], str)
        if input_nodes[1] is None:
            index = torch.arange(data[self.input_nodes[0]].num_nodes)
            input_nodes = (input_nodes[0], index)
        elif input_nodes[1].dtype == torch.bool:
            index = input_nodes[1].nonzero(as_tuple=False).view(-1)
            input_nodes = (input_nodes[0], index)

        self.data = data
        self.num_samples = num_samples
        self.input_nodes = input_nodes
        self.num_hops = max([len(v) for v in num_samples.values()])
        self.transform = transform
        self.hgt_sample = torch.ops.torch_sparse.hgt_sample

        self.colptr_dict, self.row_dict, self.perm_dict = _convert_input(data)

        super().__init__(input_nodes[1].tolist(), collate_fn=self.sample,
                         **kwargs)

    def sample(self, indices: List[int]) -> HeteroData:
        input_node_dict = {self.input_nodes[0]: torch.tensor(indices)}

        node_dict, row_dict, col_dict, edge_dict = self.hgt_sample(
            self.colptr_dict, self.row_dict, input_node_dict, self.num_samples,
            self.num_hops)

        # After sampling, we create a new `HeteroData` object holding subgraphs
        # of the original data. We do this by copying the original data, and
        # filter the required information for each node/edge type afterwards.
        data = copy.copy(self.data)

        for node_type in data.node_types:  # Filter data for each node type:
            if node_type not in node_dict:
                del data[node_type]
                continue

            n_id = node_dict[node_type]
            store = data.get_node_store(node_type)
            num_nodes = self.data.get_node_store(node_type).num_nodes
            for key, value in store.items():
                if key == 'num_nodes':
                    store[key] = n_id.numel()
                elif isinstance(value, Tensor) and value.size(0) == num_nodes:
                    store[key] = value[n_id]

        for edge_type in data.edge_types:  # Filter data for each edge type:
            rel_type = '__'.join(edge_type)

            if rel_type not in row_dict:
                del data[edge_type]
                continue

            e_id = edge_dict[rel_type]
            store = data.get_edge_store(*edge_type)
            num_edges = self.data.get_edge_store(*edge_type).num_edges
            for key, value in store.items():
                if key == 'edge_index':
                    row, col = row_dict[rel_type], col_dict[rel_type]
                    store[key] = torch.stack([row, col], dim=0)
                elif key == 'adj_t':
                    row, col = row_dict[rel_type], col_dict[rel_type]
                    weight = value.storage.value()
                    weight = weight[e_id] if weight is not None else weight
                    size = (data[edge_type[0]].num_nodes,
                            data[edge_type[-1]].num_nodes)
                    store[key] = SparseTensor(row=col, col=row, value=weight,
                                              sparse_sizes=size[::-1],
                                              is_sorted=True)
                elif isinstance(value, Tensor) and value.size(0) == num_edges:
                    if rel_type in self.perm_dict:
                        store[key] = value[self.perm_dict[rel_type]][e_id]
                    else:
                        store[key] = value[e_id]

        # We set the batch size as an attribute to the specific node type:
        data[self.input_nodes[0]].batch_size = len(indices)

        data = self.transform(data) if self.transform is not None else data

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


def _convert_input(
    data: HeteroData,
) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[str, Tensor]]:
    # The HGTSampler expects a heterogeneous graph to be stored in CSC
    # representation. We therefore iterate over each edge type and convert the
    # `edge_index`/`adj_t` tensor into `(colptr, row, perm)` tuples, where
    # `perm` denotes the re-ordering/permutation of original edge values to CSC
    # ordering.

    ind2ptr = torch.ops.torch_sparse.ind2ptr

    colptr_dict, row_dict, perm_dict = {}, {}, {}

    for edge_type in data.edge_types:
        rel_type = '__'.join(edge_type)
        store = data.get_edge_store(*edge_type)

        if 'adj_t' in store:
            colptr, row, _ = store.adj_t.csr()
            colptr_dict[rel_type] = colptr.cpu()
            row_dict[rel_type] = row.cpu()

        elif 'edge_index' in store:
            (row, col), size = store.edge_index, store.size()
            perm = (col * size[0]).add_(row).argsort()
            colptr_dict[rel_type] = ind2ptr(col[perm], size[1]).cpu()
            row_dict[rel_type] = row[perm].cpu()
            perm_dict[rel_type] = perm.cpu()

    return colptr_dict, row_dict, perm_dict
