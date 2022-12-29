import warnings
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor

from torch_geometric.data import HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.typing import EdgeType, SparseTensor
from torch_geometric.utils import coalesce, degree


@functional_transform('add_metapaths')
class AddMetaPaths(BaseTransform):
    r""" Adds additional edge types to a
    :class:`~torch_geometric.data.HeteroData` object between the source node
    type and the destination node type of a given :obj:`metapath`, as described
    in the `"Heterogenous Graph Attention Networks"
    <https://arxiv.org/abs/1903.07293>`_ paper
    (functional name: :obj:`add_metapaths`).
    Meta-path based neighbors can exploit different aspects of structure
    information in heterogeneous graphs.
    Formally, a metapath is a path of the form

    .. math::

        \mathcal{V}_1 \xrightarrow{R_1} \mathcal{V}_2 \xrightarrow{R_2} \ldots
        \xrightarrow{R_{\ell-1}} \mathcal{V}_{\ell}

    in which :math:`\mathcal{V}_i` represents node types, and :math:`R_j`
    represents the edge type connecting two node types.
    The added edge type is given by the sequential multiplication  of
    adjacency matrices along the metapath, and is added to the
    :class:`~torch_geometric.data.HeteroData` object as edge type
    :obj:`(src_node_type, "metapath_*", dst_node_type)`, where
    :obj:`src_node_type` and :obj:`dst_node_type` denote :math:`\mathcal{V}_1`
    and :math:`\mathcal{V}_{\ell}`, repectively.

    In addition, a :obj:`metapath_dict` object is added to the
    :class:`~torch_geometric.data.HeteroData` object which maps the
    metapath-based edge type to its original metapath.

    .. code-block:: python

        from torch_geometric.datasets import DBLP
        from torch_geometric.data import HeteroData
        from torch_geometric.transforms import AddMetaPaths

        data = DBLP(root)[0]
        # 4 node types: "paper", "author", "conference", and "term"
        # 6 edge types: ("paper","author"), ("author", "paper"),
        #               ("paper, "term"), ("paper", "conference"),
        #               ("term, "paper"), ("conference", "paper")

        # Add two metapaths:
        # 1. From "paper" to "paper" through "conference"
        # 2. From "author" to "conference" through "paper"
        metapaths = [[("paper", "conference"), ("conference", "paper")],
                     [("author", "paper"), ("paper", "conference")]]
        data = AddMetaPaths(metapaths)(data)

        print(data.edge_types)
        >>> [("author", "to", "paper"), ("paper", "to", "author"),
             ("paper", "to", "term"), ("paper", "to", "conference"),
             ("term", "to", "paper"), ("conference", "to", "paper"),
             ("paper", "metapath_0", "paper"),
             ("author", "metapath_1", "conference")]

        print(data.metapath_dict)
        >>> {("paper", "metapath_0", "paper"): [("paper", "conference"),
                                                ("conference", "paper")],
             ("author", "metapath_1", "conference"): [("author", "paper"),
                                                      ("paper", "conference")]}

    Args:
        metapaths (List[List[Tuple[str, str, str]]]): The metapaths described
            by a list of lists of
            :obj:`(src_node_type, rel_type, dst_node_type)` tuples.
        drop_orig_edge_types (bool, optional): If set to :obj:`True`, existing
            edge types will be dropped. (default: :obj:`False`)
        keep_same_node_type (bool, optional): If set to :obj:`True`, existing
            edge types between the same node type are not dropped even in case
            :obj:`drop_orig_edge_types` is set to :obj:`True`.
            (default: :obj:`False`)
        drop_unconnected_node_types (bool, optional): If set to :obj:`True`,
            will drop node types not connected by any edge type.
            (default: :obj:`False`)
        max_sample (int, optional): If set, will sample at maximum
            :obj:`max_sample` neighbors within metapaths. Useful in order to
            tackle very dense metapath edges. (default: :obj:`None`)
        weighted (bool, optional): If set to :obj:`True` compute weights for
            each metapath edge and store them in :obj:`edge_weight`. The weight
            of each metapath edge is computed as the number of metapaths from
            the start to the end of the metapath edge.
            (default :obj:`False`)
    """
    def __init__(
        self,
        metapaths: List[List[EdgeType]],
        drop_orig_edge_types: bool = False,
        keep_same_node_type: bool = False,
        drop_unconnected_node_types: bool = False,
        max_sample: Optional[int] = None,
        weighted: bool = False,
        **kwargs,
    ):

        if 'drop_orig_edges' in kwargs:
            warnings.warn("'drop_orig_edges' is dprecated. Use "
                          "'drop_orig_edge_types' instead")
            drop_orig_edge_types = kwargs['drop_orig_edges']

        if 'drop_unconnected_nodes' in kwargs:
            warnings.warn("'drop_unconnected_nodes' is dprecated. Use "
                          "'drop_unconnected_node_types' instead")
            drop_unconnected_node_types = kwargs['drop_unconnected_nodes']

        for path in metapaths:
            assert len(path) >= 2, f"Invalid metapath '{path}'"
            assert all([
                j[-1] == path[i + 1][0] for i, j in enumerate(path[:-1])
            ]), f"Invalid sequence of node types in '{path}'"

        self.metapaths = metapaths
        self.drop_orig_edge_types = drop_orig_edge_types
        self.keep_same_node_type = keep_same_node_type
        self.drop_unconnected_node_types = drop_unconnected_node_types
        self.max_sample = max_sample
        self.weighted = weighted

    def __call__(self, data: HeteroData) -> HeteroData:
        edge_types = data.edge_types  # save original edge types
        data.metapath_dict = {}

        for j, metapath in enumerate(self.metapaths):
            for edge_type in metapath:
                assert data._to_canonical(
                    edge_type) in edge_types, f"'{edge_type}' not present"

            edge_type = metapath[0]
            edge_weight = self._get_edge_weight(data, edge_type)
            adj1 = SparseTensor.from_edge_index(
                edge_index=data[edge_type].edge_index,
                sparse_sizes=data[edge_type].size(), edge_attr=edge_weight)

            if self.max_sample is not None:
                adj1 = self.sample_adj(adj1)

            for i, edge_type in enumerate(metapath[1:]):
                edge_weight = self._get_edge_weight(data, edge_type)
                adj2 = SparseTensor.from_edge_index(
                    edge_index=data[edge_type].edge_index,
                    sparse_sizes=data[edge_type].size(), edge_attr=edge_weight)

                adj1 = adj1 @ adj2

                if self.max_sample is not None:
                    adj1 = self.sample_adj(adj1)

            row, col, edge_weight = adj1.coo()
            new_edge_type = (metapath[0][0], f'metapath_{j}', metapath[-1][-1])
            data[new_edge_type].edge_index = torch.vstack([row, col])
            if self.weighted:
                data[new_edge_type].edge_weight = edge_weight
            data.metapath_dict[new_edge_type] = metapath

        postprocess(data, edge_types, self.drop_orig_edge_types,
                    self.keep_same_node_type, self.drop_unconnected_node_types)

        return data

    def sample_adj(self, adj: SparseTensor) -> SparseTensor:
        row, col, _ = adj.coo()
        deg = degree(row, num_nodes=adj.size(0))
        prob = (self.max_sample * (1. / deg))[row]
        mask = torch.rand_like(prob) < prob
        return adj.masked_select_nnz(mask, layout='coo')

    def _get_edge_weight(self, data: HeteroData,
                         edge_type: EdgeType) -> torch.Tensor:
        if self.weighted:
            edge_weight = data[edge_type].get('edge_weight', None)
            if edge_weight is None:
                edge_weight = torch.ones(
                    data[edge_type].num_edges,
                    device=data[edge_type].edge_index.device)
            assert edge_weight.ndim == 1
        else:
            edge_weight = None
        return edge_weight


@functional_transform('add_random_metapaths')
class AddRandomMetaPaths(BaseTransform):
    r""" Adds additional edge types similar to :class:`AddMetaPaths`.
    The key difference is that the added edge type is given by
    multiple random walks along the metapath.
    One might want to increase the number of random walks
    via :obj:`walks_per_node` to achieve competitive performance with
    :class:`AddMetaPaths`.

    Args:
        metapaths (List[List[Tuple[str, str, str]]]): The metapaths described
            by a list of lists of
            :obj:`(src_node_type, rel_type, dst_node_type)` tuples.
        drop_orig_edge_types (bool, optional): If set to :obj:`True`, existing
            edge types will be dropped. (default: :obj:`False`)
        keep_same_node_type (bool, optional): If set to :obj:`True`, existing
            edge types between the same node type are not dropped even in case
            :obj:`drop_orig_edge_types` is set to :obj:`True`.
            (default: :obj:`False`)
        drop_unconnected_node_types (bool, optional): If set to :obj:`True`,
            will drop node types not connected by any edge type.
            (default: :obj:`False`)
        walks_per_node (int, List[int], optional): The number of random walks
            for each starting node in a metapth. (default: :obj:`1`)
        sample_ratio (float, optional): The ratio of source nodes to start
            random walks from. (default: :obj:`1.0`)
    """
    def __init__(
        self,
        metapaths: List[List[EdgeType]],
        drop_orig_edge_types: bool = False,
        keep_same_node_type: bool = False,
        drop_unconnected_node_types: bool = False,
        walks_per_node: Union[int, List[int]] = 1,
        sample_ratio: float = 1.0,
    ):

        for path in metapaths:
            assert len(path) >= 2, f"Invalid metapath '{path}'"
            assert all([
                j[-1] == path[i + 1][0] for i, j in enumerate(path[:-1])
            ]), f"Invalid sequence of node types in '{path}'"

        self.metapaths = metapaths
        self.drop_orig_edge_types = drop_orig_edge_types
        self.keep_same_node_type = keep_same_node_type
        self.drop_unconnected_node_types = drop_unconnected_node_types
        self.sample_ratio = sample_ratio
        if isinstance(walks_per_node, int):
            walks_per_node = [walks_per_node] * len(metapaths)
        assert len(walks_per_node) == len(metapaths)
        self.walks_per_node = walks_per_node

    def __call__(self, data: HeteroData) -> HeteroData:
        edge_types = data.edge_types  # save original edge types
        data.metapath_dict = {}

        for j, metapath in enumerate(self.metapaths):
            for edge_type in metapath:
                assert data._to_canonical(
                    edge_type) in edge_types, f"'{edge_type}' not present"

            src_node = metapath[0][0]
            num_nodes = data[src_node].num_nodes
            num_starts = round(num_nodes * self.sample_ratio)
            row = start = torch.randperm(num_nodes)[:num_starts].repeat(
                self.walks_per_node[j])

            for i, edge_type in enumerate(metapath):
                edge_index = data[edge_type].edge_index
                adj = SparseTensor(row=edge_index[0], col=edge_index[1],
                                   sparse_sizes=data[edge_type].size())
                col, mask = self.sample(adj, start)
                row, col = row[mask], col[mask]
                start = col

            new_edge_type = (metapath[0][0], f'metapath_{j}', metapath[-1][-1])
            data[new_edge_type].edge_index = coalesce(torch.vstack([row, col]))
            data.metapath_dict[new_edge_type] = metapath

        postprocess(data, edge_types, self.drop_orig_edge_types,
                    self.keep_same_node_type, self.drop_unconnected_node_types)

        return data

    @staticmethod
    def sample(adj: SparseTensor, subset: Tensor) -> Tuple[Tensor, Tensor]:
        """Sample a neighbor form `adj` for each node in `subset`"""

        rowcount = adj.storage.rowcount()[subset]
        mask = rowcount > 0
        offset = torch.zeros_like(subset)
        offset[mask] = adj.storage.rowptr()[subset[mask]]

        rand = torch.rand((rowcount.size(0), 1), device=subset.device)
        rand.mul_(rowcount.to(rand.dtype).view(-1, 1))
        rand = rand.to(torch.long)
        rand.add_(offset.view(-1, 1))
        col = adj.storage.col()[rand].squeeze()
        return col, mask

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'sample_ratio={self.sample_ratio}, '
                f'walks_per_node={self.walks_per_node})')


def postprocess(data: HeteroData, edge_types: List[EdgeType],
                drop_orig_edge_types: bool, keep_same_node_type: bool,
                drop_unconnected_node_types: bool):

    if drop_orig_edge_types:
        for i in edge_types:
            if keep_same_node_type and i[0] == i[-1]:
                continue
            else:
                del data[i]

    # remove nodes not connected by any edge type.
    if drop_unconnected_node_types:
        new_edge_types = data.edge_types
        node_types = data.node_types
        connected_nodes = set()
        for i in new_edge_types:
            connected_nodes.add(i[0])
            connected_nodes.add(i[-1])
        for node in node_types:
            if node not in connected_nodes:
                del data[node]
