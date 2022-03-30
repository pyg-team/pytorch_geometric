from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

from torch_geometric.data import HeteroData
from torch_geometric.loader.utils import filter_hetero_data, to_hetero_csc
from torch_geometric.typing import NodeType


class HGTLoader(torch.utils.data.DataLoader):
    r"""The Heterogeneous Graph Sampler from the `"Heterogeneous Graph
    Transformer" <https://arxiv.org/abs/2003.01332>`_ paper.
    This loader allows for mini-batch training of GNNs on large-scale graphs
    where full-batch training is not feasible.

    :class:`~torch_geometric.data.HGTLoader` tries to (1) keep a similar
    number of nodes and edges for each type and (2) keep the sampled sub-graph
    dense to minimize the information loss and reduce the sample variance.

    Methodically, :class:`~torch_geometric.data.HGTLoader` keeps track of a
    node budget for each node type, which is then used to determine the
    sampling probability of a node.
    In particular, the probability of sampling a node is determined by the
    number of connections to already sampled nodes and their node degrees.
    With this, :class:`~torch_geometric.data.HGTLoader` will sample a fixed
    amount of neighbors for each node type in each iteration, as given by the
    :obj:`num_samples` argument.

    Sampled nodes are sorted based on the order in which they were sampled.
    In particular, the first :obj:`batch_size` nodes represent the set of
    original mini-batch nodes.

    .. note::

        For an example of using :class:`~torch_geometric.data.HGTLoader`, see
        `examples/hetero/to_hetero_mag.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        hetero/to_hetero_mag.py>`_.

    .. code-block:: python

        from torch_geometric.loader import HGTLoader
        from torch_geometric.datasets import OGB_MAG

        hetero_data = OGB_MAG(path)[0]

        loader = HGTLoader(
            hetero_data,
            # Sample 512 nodes per type and per iteration for 4 iterations
            num_samples={key: [512] * 4 for key in hetero_data.node_types},
            # Use a batch size of 128 for sampling training nodes of type paper
            batch_size=128,
            input_nodes=('paper': hetero_data['paper'].train_mask),
        )

        sampled_hetero_data = next(iter(loader))
        print(sampled_data.batch_size)
        >>> 128

    Args:
        data (torch_geometric.data.HeteroData): The
            :class:`~torch_geometric.data.HeteroData` graph data object.
        num_samples (List[int] or Dict[str, List[int]]): The number of nodes to
            sample in each iteration and for each node type.
            If given as a list, will sample the same amount of nodes for each
            node type.
        input_nodes (str or Tuple[str, torch.Tensor]): The indices of nodes for
            which neighbors are sampled to create mini-batches.
            Needs to be passed as a tuple that holds the node type and
            corresponding node indices.
            Node indices need to be either given as a :obj:`torch.LongTensor`
            or :obj:`torch.BoolTensor`.
            If node indices are set to :obj:`None`, all nodes of this specific
            type will be considered.
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
        input_nodes: Union[NodeType, Tuple[NodeType, Optional[Tensor]]],
        transform: Callable = None,
        **kwargs,
    ):
        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']

        if isinstance(num_samples, (list, tuple)):
            num_samples = {key: num_samples for key in data.node_types}

        if isinstance(input_nodes, str):
            input_nodes = (input_nodes, None)
        assert isinstance(input_nodes, (list, tuple))
        assert len(input_nodes) == 2
        assert isinstance(input_nodes[0], str)
        if input_nodes[1] is None:
            index = torch.arange(data[input_nodes[0]].num_nodes)
            input_nodes = (input_nodes[0], index)
        elif input_nodes[1].dtype == torch.bool:
            index = input_nodes[1].nonzero(as_tuple=False).view(-1)
            input_nodes = (input_nodes[0], index)

        self.data = data
        self.num_samples = num_samples
        self.input_nodes = input_nodes
        self.num_hops = max([len(v) for v in num_samples.values()])
        self.transform = transform
        self.sample_fn = torch.ops.torch_sparse.hgt_sample

        # Convert the graph data into a suitable format for sampling.
        # NOTE: Since C++ cannot take dictionaries with tuples as key as
        # input, edge type triplets are converted into single strings.
        self.colptr_dict, self.row_dict, self.perm_dict = to_hetero_csc(
            data, device='cpu')

        super().__init__(input_nodes[1].tolist(), collate_fn=self.sample,
                         **kwargs)

    def sample(self, indices: List[int]) -> HeteroData:
        input_node_dict = {self.input_nodes[0]: torch.tensor(indices)}
        node_dict, row_dict, col_dict, edge_dict = self.sample_fn(
            self.colptr_dict,
            self.row_dict,
            input_node_dict,
            self.num_samples,
            self.num_hops,
        )

        data = filter_hetero_data(self.data, node_dict, row_dict, col_dict,
                                  edge_dict, self.perm_dict)
        data[self.input_nodes[0]].batch_size = len(indices)

        return data if self.transform is None else self.transform(data)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
