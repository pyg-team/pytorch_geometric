import torch
import re
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_sparse import SparseTensor

class RootedSubgraphsData(Data):
    r""" A data object describing a homogeneous graph together with each node's rooted subgraph. 
    It contains several additional propreties that hold the information of all nodes' rooted subgraphs.
    Assume the data represents a graph with :math:'N' nodes and :math:'M' edges, also each node 
    :math:'i\in \[N\]' has a rooted subgraph with :math:'N_i' nodes and :math:'M_i' edges.
    
    Additional Properties:
        subgraphs_nodes_mapper (LongTensor): map each node in rooted subgraphs to a node in the original graph.
            Size: :math:'\sum_{i=1}^{N}N_i x 1'
        subgraphs_edges_mapper (LongTensor): map each edge in rooted subgraphs to a edge in the original graph.
            Size: :math:'\sum_{i=1}^{N}M_i x 1'
        subgraphs_batch: map each node in rooted subgraphs to its corresponding rooted subgraph index. 
            Size: :math:'\sum_{i=1}^{N}N_i x 1'
        combined_rooted_subgraphs: edge_index of a giant graph which represents a stacking of all rooted subgraphs. 
            Size: :math:'2 x \sum_{i=1}^{N}M_i'
        
    The class works as a wrapper for the data with these properties, and automatically handles mini batching for
    them. 
    
    """
    def __inc__(self, key, value, *args, **kwargs):
        num_nodes = self.num_nodes
        num_edges = self.edge_index.size(-1)
        if bool(re.search('(combined_subgraphs)', key)):
            return getattr(
                self, key[:-len('combined_subgraphs')] +
                'subgraphs_nodes_mapper').size(0)
        elif bool(re.search('(subgraphs_batch)', key)):
            # should use number of subgraphs or number of supernodes.
            return 1 + getattr(self, key)[-1]
        elif bool(re.search('(nodes_mapper)|(selected_supernodes)', key)):
            return num_nodes
        elif bool(re.search('(edges_mapper)', key)):
            # batched_edge_attr[subgraphs_edges_mapper] shoud be batched_combined_subgraphs_edge_attr
            return num_edges
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if bool(re.search('(combined_subgraphs)', key)):
            return -1
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)

class RootedSubgraphs(BaseTransform):
    r"""
    Base class for rooted subgraphs.
    The object transform a Data object to RootedSubgraphsData object. 
    """
    def __init__(self):
        super().__init__()

    def extract_subgraphs(self, data: Data):
        r""" For a input graph with N nodes, extract a rooted subgraph for every node in the graph.
        Return:
            subgraphs_nodes_mask: N x N dense mask matrix, i-th row indicates the rooted subgraph of 
                node i. 
            hop_indicator: N x N dense matrix, i-th row indicates the distance from other nodes to 
                node i.
        """
        raise NotImplementedError

    def __call__(self, data: Data) -> Data:
        subgraphs_nodes_mask, hop_indicator_dense = self.extract_subgraphs(data)
        subgraphs_edges_mask = subgraphs_nodes_mask[:, data.edge_index[0]] & \
                               subgraphs_nodes_mask[:, data.edge_index[1]]  # N x E dense mask matrix

        subgraphs_nodes, subgraphs_edges, hop_indicator = to_sparse(subgraphs_nodes_mask, 
                                                                    subgraphs_edges_mask, 
                                                                    hop_indicator_dense)
        combined_subgraphs = combine_subgraphs(data.edge_index,
                                               subgraphs_nodes,
                                               subgraphs_edges,
                                               num_selected=data.num_nodes,
                                               num_nodes=data.num_nodes)

        data = RootedSubgraphsData(**{k: v for k, v in data})
        data.subgraphs_batch = subgraphs_nodes[0]
        data.subgraphs_nodes_mapper = subgraphs_nodes[1]
        data.subgraphs_edges_mapper = subgraphs_edges[1]
        data.combined_subgraphs = combined_subgraphs
        data.hop_indicator = hop_indicator
        data.__num_nodes__ = data.num_nodes
        return data
    
class RootedEgoNets(RootedSubgraphs):
    """ Record rooted k-hop Egonet for each node in the graph.
    From the `"From Stars to Subgraphs: Uplifting Any GNN with Local Structure Awareness"
    <https://arxiv.org/pdf/2110.03753.pdf>`_ paper

    Args:
        hops (int): k for k-hop Egonet. 
    """
    def __init__(self, hops: int):
        super().__init__()
        self.num_hops = hops

    def extract_subgraphs(self, data: Data):
        # return k-hop subgraphs for all nodes in the graph
        row, col = data.edge_index
        sparse_adj = SparseTensor(row=row, col=col, sparse_sizes=(data.num_nodes, data.num_nodes))
        hop_mask = sparse_adj.to_dense() > 0
        hop_indicator = torch.eye(data.num_nodes, dtype=torch.bool, device=data.edge_index.device) - 1

        for i in range(self.num_hops):
            hop_indicator[(hop_indicator == -1) & hop_mask] = i + 1
            hop_mask = sparse_adj.matmul(hop_mask.float()) > 0

        hop_indicator = hop_indicator.T  # N x N
        node_mask = (hop_indicator >= 0)  # N x N dense mask matrix

        return node_mask, hop_indicator


    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.hops})'

from torch_cluster import random_walk
class RootedRWSubgraphs(RootedSubgraphs):
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
    def __init__(self, walk_length: int=0, p: float=1., q: float=1., repeat: int=1, return_hops: bool=True, max_hops: int=10):
        super().__init__()
        self.walk_length = walk_length
        self.p = p
        self.q = q
        self.repeat = repeat
        self.max_hops = max_hops
        self.return_hops = return_hops

    def extract_subgraphs(self, data: Data):
        row, col = data.edge_index
        num_nodes = data.num_nodes
        start = torch.arange(num_nodes, device=row.device)
        walks = [random_walk(row, col, 
                             start=start, 
                             walk_length=self.walk_length, 
                             p=self.p, 
                             q=self.q,
                             num_nodes=num_nodes) 
            for _ in range(self.repeat)
        ]
        walk = torch.cat(walks, dim=-1)
        node_mask = row.new_empty((num_nodes, num_nodes), dtype=torch.bool)
        # print(walk.shape)
        node_mask.fill_(False)
        node_mask[start.repeat_interleave((self.walk_length + 1) * self.repeat),
                walk.reshape(-1)] = True

        if self.return_hops:  # this is fast enough
            sparse_adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
            hop_mask = sparse_adj.to_dense() > 0
            hop_indicator = torch.eye(data.num_nodes, dtype=torch.bool, device=data.edge_index.device) - 1

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
    subgraphs_nodes = node_mask.nonzero().T
    subgraphs_edges = edge_mask.nonzero().T
    if hop_indicator is not None:
        hop_indicator = hop_indicator[subgraphs_nodes[0], subgraphs_nodes[1]]
    return subgraphs_nodes, subgraphs_edges,

def combine_subgraphs(edge_index, subgraphs_nodes, subgraphs_edges,
                      num_selected=None, num_nodes=None):
    if num_selected is None:
        num_selected = subgraphs_nodes[0][-1] + 1
    if num_nodes is None:
        num_nodes = subgraphs_nodes[1].max() + 1

    combined_subgraphs = edge_index[:, subgraphs_edges[1]]
    node_label_mapper = edge_index.new_full((num_selected, num_nodes), -1)
    node_label_mapper[subgraphs_nodes[0], subgraphs_nodes[1]] = torch.arange(
        len(subgraphs_nodes[1]))
    node_label_mapper = node_label_mapper.reshape(-1)

    inc = torch.arange(num_selected) * num_nodes
    combined_subgraphs += inc[subgraphs_edges[0]]
    combined_subgraphs = node_label_mapper[combined_subgraphs]
    return combined_subgraphs
