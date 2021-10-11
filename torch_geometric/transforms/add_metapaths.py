from copy import copy
from typing import List

import torch

from torch_geometric.typing import EdgeType
from torch_geometric.data import HeteroData
from torch_sparse.spspmm import spspmm
from torch_geometric.utils.num_nodes import maybe_num_nodes_dict
from torch_geometric.transforms import BaseTransform


class AddMetaPaths(BaseTransform):
    r""" Adds edges between nodes connected by
    :obj:`metapaths`, to a given heterogenous graph.
    Metapaths are defined in
    `"Heterogenous Graph Attention Networks"
    <https://arxiv.org/abs/1903.07293>`_ paper.

    .. code-block:: python

        from torch_geometric.data import HeteroData
        from torch_geometric.transforms import AddMetaPaths

        dblp = HeteroData()
        dblp['paper'].x = ...
        dblp['author'].x = ...
        dblp['conference'].x = ...
        dblp['paper', 'cites', 'paper'].edge_index = ...
        dblp['paper', 'author'].edge_index = ...
        dblp['author', 'paper'].edge_index = ...
        dblp['conference', 'paper'].edge_index = ...
        dblp['paper', 'conference'].edge_index = ...

        # Add two metapaths
        # 1. from 'paper' to 'paper' through 'conference'.
        # 2. from 'author' to 'conference' through 'papers'.
        # New edges are of type ('paper', 'metapath_0', 'paper')
        # and ('author', 'metapath_1', 'conference')
        metapaths = [[('paper', 'to', 'conference'),
                      ('conference', 'to', 'paper')],
                    [('author', 'to', 'paper'), ('paper', 'to', 'conference')]]
        dblp_meta = AddMetaPaths(metapaths)(dblp.clone())
        print(dblp_meta.edge_types)
        >>> [('paper', 'cites', 'paper'), ('paper', 'to', 'author'),
            ('author', 'to', 'paper'), ('conference', 'to', 'paper'),
            ('paper', 'to', 'conference'), ('paper', 'metapath_0', 'paper'),
            ('author', 'metapath_1', 'conference')]

        # info on added metapaths
        print(dblp_meta.metapaths)
        >>> [[('paper', 'to', 'conference'), ('conference', 'to', 'paper')],
             [('author', 'to', 'paper'), ('paper', 'to', 'conference')]]

        # Add metapaths, and drop all other edges except
        # edges between same node type.
        dblp_meta = AddMetaPaths(metapaths, drop_orig_edges=True,
                                 keep_same_node_type=True)(dblp.clone())
        print(dblp_meta.edge_types)
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
                 keep_same_node_type: bool = False) -> None:
        super().__init__()

        for path in metapaths:
            assert len(path) >= 2, f'invalid metapath {path}'
            assert all([len(i) >= 2
                        for i in path]), f'invalid edge type in {path}'
            assert all([
                j[-1] == path[i + 1][0] for i, j in enumerate(path[:-1])
            ]), f'invalid sequence of nodes {path}'

        self.metapaths = metapaths
        self.drop_orig_edges = drop_orig_edges
        self.keep_same_node_type = keep_same_node_type

    def __call__(self, data: HeteroData) -> HeteroData:

        num_nodes_dict = maybe_num_nodes_dict(data.edge_index_dict)
        for i in data.node_types:
            if 'x' in data[i]:
                num_nodes_dict[i] = data[i].x.size(0)

        edge_types = copy(data.edge_types)  # save original edge types
        for j, metapath in enumerate(self.metapaths):

            for path in metapath:
                assert path in edge_types, (f"{path} edge not present."
                                            "Try adding rel_type 'to'")

            new_edge_type = f'metapath_{j}'
            start_node = metapath[0][0]
            end_node = metapath[-1][-1]
            new_edge_index = data[metapath[0]].edge_index
            m = num_nodes_dict[metapath[0][0]]

            for i, path in enumerate(metapath[1:]):
                k = num_nodes_dict[path[0]]
                n = num_nodes_dict[path[2]]
                edge_index = data[path].edge_index
                valueA = new_edge_index.new_ones((new_edge_index.size(1), ),
                                                 dtype=torch.float)
                valueB = edge_index.new_ones((edge_index.size(1), ),
                                             dtype=torch.float)
                new_edge_index, _ = spspmm(new_edge_index, valueA, edge_index,
                                           valueB, m, k, n, coalesced=True)
                data[(start_node, new_edge_type,
                      end_node)].edge_index = new_edge_index

        if self.drop_orig_edges:
            for i in edge_types:
                if self.keep_same_node_type and i[0] == i[-1]:
                    continue
                else:
                    del data[i]

        # add metapaths for later reference
        data.metapaths = self.metapaths
        return data
