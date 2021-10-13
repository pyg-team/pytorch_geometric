from typing import List

import torch

from torch_geometric.typing import EdgeType
from torch_geometric.data import HeteroData
from torch_sparse import SparseTensor
from torch_geometric.transforms import BaseTransform


class AddMetaPaths(BaseTransform):
    r""" Adds edges to a :obj:`HeteroData` object between nodes connected
    by given :obj:`metapaths`. As described in `"Heterogenous Graph Attention
    Networks" <https://arxiv.org/abs/1903.07293>`_ paper a metapath is a path
    of the form

    .. math::

        V_1 \xrightarrow{R1} V_2 \xrightarrow{R2}...\xrightarrow{R_{l-1}} V_l

        R= R_1\circ R_2\circ...\circ R_{l-1}

    Here :math:`R` is the composite relation between node types
    :math:`V_1` and :math:`V_l`.

    The added edge for the :math:`i^{th}` metapath is of type
    :obj:`(src_node_type, "metapath_i", dst_node_type)` where
    :obj:`src_node_type` and :obj:`dst_node_type` are :math:`V_1`
    and :math:`V_l` repectively. A :obj:`metapaths_dict` is
    added to the :obj:`HeteroData` object. With key
    :obj:`(src_node_type, "metapath_i", dst_node_type)` and value
    :obj:`metapaths[i]`.

    .. code-block:: python

        from torch_geometric.data import HeteroData
        from torch_geometric.transforms import AddMetaPaths
        from torch_geometric.datasets import DBLP

        data = DBLP(root)[0]
        # 4 node types: "paper", "author", "conference", and "term"
        # 6 edge types: ("paper","author"), ("author", "paper"),
        #               ("paper, "term"),("paper", "conference"),
        #               ("term, "paper"), ("conference", "paper")

        # Add two metapaths
        # 1. from 'paper' to 'paper' through 'conference'.
        # 2. from 'author' to 'conference' through 'paper'.
        # New edges are of type ('paper', 'metapath_0', 'paper')
        # and ('author', 'metapath_1', 'conference')
        metapaths = [[('paper','conference'),('conference','paper')],
                    [('author','paper'), ('paper','conference')]]
        data_meta = AddMetaPaths(metapaths)(data.clone())
        print(data_meta.edge_types)
        >>> [('author', 'to', 'paper'),('paper', 'to', 'author'),
             ('paper', 'to', 'term'),('paper', 'to', 'conference'),
             ('term', 'to', 'paper'),('conference', 'to', 'paper'),
             ('paper', 'metapath_0', 'paper'),
             ('author', 'metapath_1', 'conference')]

        # info on added edge types
        print(data_meta.metapaths_dict)
        >>> {('paper', 'metapath_0', 'paper'): [('paper', 'conference'),
                                                ('conference', 'paper')],
             ('author', 'metapath_1', 'conference'): [('author', 'paper'),
                                                      ('paper', 'conference')]}

        # add a new edge type between same node types
        data['paper', 'cites', 'paper'].edge_index = ...

        # Add metapaths, and drop all other edges except
        # edges between same node type.
        data_meta = AddMetaPaths(metapaths, drop_orig_edges=True,
                                 keep_same_node_type=True)(data.clone())
        print(data_meta.edge_types)
        >>> [('paper', 'cites', 'paper'), ('paper', 'metapath_0', 'paper'),
            ('author', 'metapath_1', 'conference')]

    Args:
        metapaths(List[List[Tuple[str, str, str]]]): The metapaths described as
            a list  of list of :obj:`(src_node_type, rel_type, dst_node_type)`
            tuples.
        drop_orig_edges(bool, optional): If set to :obj:`True`,
            existing edges are dropped. (default: :obj:`False`)
        keep_same_node_type(bool, optional): If set to :obj:`True` then
            existing edges between the same node type are not dropped even
            if :obj:`drop_orig_edges` is set to :obj:`True`.
            (default: :obj:`False`)

    """
    def __init__(self, metapaths: List[List[EdgeType]],
                 drop_orig_edges: bool = False,
                 keep_same_node_type: bool = False):

        for path in metapaths:
            assert len(path) >= 2, f'invalid metapath {path}'
            assert all([
                j[-1] == path[i + 1][0] for i, j in enumerate(path[:-1])
            ]), f'invalid sequence of nodes {path}'

        self.metapaths = metapaths
        self.drop_orig_edges = drop_orig_edges
        self.keep_same_node_type = keep_same_node_type

    def __call__(self, data: HeteroData) -> HeteroData:

        edge_types = data.edge_types  # save original edge types
        data.metapaths_dict = {}

        for j, metapath in enumerate(self.metapaths):

            for edge_type in metapath:
                assert data._to_canonical(
                    edge_type) in edge_types, f"{edge_type} edge not present."

            edge_type = metapath[0]
            adj1 = SparseTensor.from_edge_index(
                edge_index=data[edge_type].edge_index,
                sparse_sizes=data[edge_type].size())

            for i, edge_type in enumerate(metapath[1:]):
                adj2 = SparseTensor.from_edge_index(
                    edge_index=data[edge_type].edge_index,
                    sparse_sizes=data[edge_type].size())
                adj1 = adj1 @ adj2

            row, col, _ = adj1.coo()
            new_edge_type = (metapath[0][0], f'metapath_{j}', metapath[-1][-1])
            data[new_edge_type].edge_index = torch.vstack([row, col])
            data.metapaths_dict[new_edge_type] = metapath

        if self.drop_orig_edges:
            for i in edge_types:
                if self.keep_same_node_type and i[0] == i[-1]:
                    continue
                else:
                    del data[i]

        return data
