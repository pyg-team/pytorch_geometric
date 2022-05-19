import torch
from typing import Optional, Tuple
from torch_sparse import SparseTensor

from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class RootedSubgraphData(Data):
    r"""A data object describing a homogeneous graph together with each node's rooted subgraph.
    It contains several additional properties that hold the information of all nodes' rooted subgraphs.
    Assume the data represents a graph with :math:`N` nodes and :math:`M` edges, also each node
    :math:`i\in \[N\]` has a rooted subgraph with :math:`N_i` nodes and :math:`M_i` edges.

    Additional Properties:
        subgraph_nodes_mapper (LongTensor): map each node in rooted subgraph to a node in the original graph.
            Size: :math:`\sum_{i=1}^{N}N_i x 1`
        subgraph_edges_mapper (LongTensor): map each edge in rooted subgraph to a edge in the original graph.
            Size: :math:`\sum_{i=1}^{N}M_i x 1`
        subgraph_batch: map each node in rooted subgraph to its corresponding rooted subgraph index.
            Size: :math:`\sum_{i=1}^{N}N_i x 1`
        combined_rooted_subgraph: edge_index of a giant graph which represents a stacking of all rooted subgraphs.
            Size: :math:`2 x \sum_{i=1}^{N}M_i`

    The class works as a wrapper for the data with these properties, and automatically handles mini batching for
    them.

    """
    def __inc__(self, key, value, *args, **kwargs):
        num_nodes = self.num_nodes
        num_edges = self.edge_index.size(-1)
        if 'combined_subgraph' in key:
            return getattr(
                self, key[:-len('combined_subgraph')] +
                'subgraph_nodes_mapper').size(0)
        elif 'subgraph_batch' in key:
            # should use number of subgraphs or number of supernodes.
            return 1 + getattr(self, key)[-1]
        elif 'nodes_mapper' in key or 'selected_supernodes' in key:
            return num_nodes
        elif 'edges_mapper' in key:
            # batched_edge_attr[subgraph_edges_mapper] shoud be batched_combined_subgraph_edge_attr
            return num_edges
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if 'combined_subgraph' in key:
            return -1
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)

    def rooted_subgraph(self, node_index: int) -> Data:
        """
        Returns a :obj:`Data` object representing the rooted subgraph at :obj:`node_index`.
        """
        nodes = self.subgraph_nodes_mapper[torch.where(self.subgraph_batch == node_index)]
        return self.subgraph(nodes)


class RootedSubgraph(BaseTransform):
    r"""
    Base class for rooted subgraph.
    The object transforms a Data object to RootedSubgraphData object.
    """

    def extract_subgraphs(self, data: Data) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r""" For input graph with N nodes, extract a rooted subgraph for every node in the graph.

        Returns: a tuple containing
            subgraph_nodes_mask: N x N dense mask matrix, i-th row indicates the rooted subgraph of
                node i.
            hop_indicator (optional): N x N dense matrix, i-th row indicates the distance from other nodes to
                node i.
        """
        raise NotImplementedError

    def __call__(self, data: Data) -> Data:
        subgraph_nodes_mask, hop_indicator_dense = self.extract_subgraphs(
            data)
        subgraph_edges_mask = subgraph_nodes_mask[:, data.edge_index[0]] & \
                               subgraph_nodes_mask[:, data.edge_index[1]]  # N x E dense mask matrix
        subgraph_nodes, subgraph_edges, hop_indicator = to_sparse(
            subgraph_nodes_mask, subgraph_edges_mask, hop_indicator_dense)

        combined_subgraph = combine_subgraph(data.edge_index,
                                             subgraph_nodes,
                                             subgraph_edges,
                                             num_selected=data.num_nodes,
                                             num_nodes=data.num_nodes)

        data = RootedSubgraphData(**{k: v for k, v in data})
        data.subgraph_batch = subgraph_nodes[0]
        data.subgraph_nodes_mapper = subgraph_nodes[1]
        data.subgraph_edges_mapper = subgraph_edges[1]
        data.combined_subgraph = combined_subgraph
        data.hop_indicator = hop_indicator
        return data


class RootedEgoNets(RootedSubgraph):
    """ Record rooted k-hop Egonet for each node in the graph.
    From the `"From Stars to Subgraphs: Uplifting Any GNN with Local Structure Awareness"
    <https://arxiv.org/pdf/2110.03753.pdf>`_ paper

    Args:
        num_hops (int): k for k-hop Egonet.
    """
    def __init__(self, num_hops: int):
        super().__init__()
        self.num_hops = num_hops

    def extract_subgraphs(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        # return k-hop subgraph for all nodes in the graph
        row, col = data.edge_index
        sparse_adj = SparseTensor(
            row=row, col=col, sparse_sizes=(data.num_nodes, data.num_nodes))
        hop_mask = sparse_adj.to_dense() > 0
        hop_indicator = torch.eye(data.num_nodes, dtype=torch.int,
                                  device=data.edge_index.device) - 1
        for i in range(self.num_hops):
            hop_indicator[(hop_indicator == -1) & hop_mask] = i + 1
            hop_mask = sparse_adj.matmul(hop_mask.float()) > 0

        node_mask = (hop_indicator >= 0)  # N x N dense mask matrix
        return node_mask, hop_indicator

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.num_hops})'


class RootedRWSubgraph(RootedSubgraph):
    """ Record rooted random-walk based subgraph for each node in the graph.
    From the `"From Stars to Subgraphs: Uplifting Any GNN with Local Structure Awareness"
    <https://arxiv.org/pdf/2110.03753.pdf>`_ paper

    Args:
        walk_length (int, optional): the length of random walk. When it is 0 use k-hop Egonet.
        p (float, optional): parameters of node2vec's random walk.
        q (float, optional): parameters of node2vec's random walk.
        repeat (int, optional): times of repeating the random walk to reduce randomness.

        return_hops (bool, optional): whether return distance to centroid feature.
        max_hops (int, optional): if return hops, the maximum D2C.
    """
    def __init__(self, walk_length: int = 0, p: float = 1., q: float = 1.,
                 repeat: int = 1, return_hops: bool = True,
                 max_hops: int = 10):
        super().__init__()
        self.walk_length = walk_length
        self.p = p
        self.q = q
        self.repeat = repeat
        self.max_hops = max_hops
        self.return_hops = return_hops

    def extract_subgraphs(self, data: Data) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        from torch_cluster import random_walk

        row, col = data.edge_index
        num_nodes = data.num_nodes
        start = torch.arange(num_nodes, device=row.device)
        walks = [
            random_walk(row, col, start=start, walk_length=self.walk_length,
                        p=self.p, q=self.q, num_nodes=num_nodes)
            for _ in range(self.repeat)
        ]
        walk = torch.cat(walks, dim=-1)
        node_mask = row.new_empty((num_nodes, num_nodes), dtype=torch.bool)
        node_mask.fill_(False)
        node_mask[start.repeat_interleave(
            (self.walk_length + 1) * self.repeat),
                  walk.reshape(-1)] = True

        if self.return_hops:  # this is fast enough
            sparse_adj = SparseTensor(row=row, col=col,
                                      sparse_sizes=(num_nodes, num_nodes))
            hop_mask = sparse_adj.to_dense() > 0
            hop_indicator = torch.eye(data.num_nodes, dtype=torch.int,
                                      device=data.edge_index.device) - 1

            for i in range(self.max_hops):
                hop_indicator[(hop_indicator == -1) & hop_mask] = i + 1
                hop_mask = sparse_adj.matmul(hop_mask.float()) > 0
                if hop_indicator[node_mask].min() != -1:
                    break
            return node_mask, hop_indicator
        return node_mask, None

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(l{self.walk_length}-p{self.p}-q{self.q}-r{self.repeat})'


def to_sparse(node_mask, edge_mask, hop_indicator):
    subgraph_nodes = node_mask.nonzero().T
    subgraph_edges = edge_mask.nonzero().T
    if hop_indicator is not None:
        hop_indicator = hop_indicator[subgraph_nodes[0], subgraph_nodes[1]]
    return subgraph_nodes, subgraph_edges, hop_indicator


def combine_subgraph(edge_index, subgraph_nodes, subgraph_edges,
                     num_selected=None, num_nodes=None):
    if num_selected is None:
        num_selected = subgraph_nodes[0][-1] + 1
    if num_nodes is None:
        num_nodes = subgraph_nodes[1].max() + 1

    combined_subgraph = edge_index[:, subgraph_edges[1]]
    node_label_mapper = edge_index.new_full((num_selected, num_nodes), -1)
    node_label_mapper[subgraph_nodes[0], subgraph_nodes[1]] = torch.arange(
        len(subgraph_nodes[1]))
    node_label_mapper = node_label_mapper.reshape(-1)

    inc = torch.arange(num_selected) * num_nodes
    combined_subgraph += inc[subgraph_edges[0]]
    combined_subgraph = node_label_mapper[combined_subgraph]
    return combined_subgraph
