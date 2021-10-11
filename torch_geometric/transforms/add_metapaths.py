from copy import deepcopy
from typing import List

import torch

from torch_geometric.typing import EdgeType
from torch_geometric.data import HeteroData
from torch_sparse.spspmm import spspmm
from torch_geometric.utils.num_nodes import maybe_num_nodes_dict
from torch_geometric.transforms import BaseTransform


class AddMetaPaths(BaseTransform):
    def __init__(self, metapaths: List[List[EdgeType]],
                 drop_orig_edges: bool = False,
                 keep_same_node_type: bool = False,
                 drop_nodes: bool = False) -> None:
        super().__init__()

        for path in metapaths:
            assert len(path) >= 2, f'invalid metapath {path}'
            assert all([len(i) == 3
                        for i in path]), f'invalid edge type in {path}'
            assert all([
                j[-1] == path[i + 1][0] for i, j in enumerate(path[:-1])
            ]), f'invalid sequence of nodes {path}'

        self.metapaths = metapaths
        self.drop_orig_edges = drop_orig_edges
        self.keep_same_node_type = keep_same_node_type

    def __call__(self, data: HeteroData):

        num_nodes_dict = maybe_num_nodes_dict(data.edge_index_dict)
        for i in data.node_types:
            if 'x' in data[i]:
                num_nodes_dict[i] = data[i].x.size(0)

        edge_types = deepcopy(data.edge_types)  # save original edge types
        for j, metapath in enumerate(self.metapaths):

            for path in metapath:
                assert path in edge_types, f'{path} edge not present'

            new_edge_type = f'metapath_{j}'
            start_node = metapath[0][0]
            end_node = metapath[-1][-1]
            new_edge_index = data[metapath[0]].edge_index
            m = num_nodes_dict[metapath[0][0]]

            for i, path in enumerate(metapath[1:]):
                k = num_nodes_dict[path[0]]
                n = num_nodes_dict[path[2]]
                edge_index = data[path].edge_index
                valueA = torch.ones(new_edge_index.shape[1])
                valueB = torch.ones(edge_index.shape[1])
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
