import torch
from torch_sparse import SparseTensor

from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class RootedSubgraphData(Data):
    r"""A data object describing a homogeneous graph together with each node's
    rooted subgraph. It contains several additional properties that hold the
    information to map to batch of every node's rooted subgraph.

    Additional Properties:
        n_id (Tensor): index of nodes includes in combined rooted subgraph.
        e_id (Tensor): index of edges included in rcombined ooted subgraph.
        n_sub_batch (Tensor): batch index for nodes in rooted subgraph index.
        e_sub_batch (Tensor): batch index for edges in rooted subgraph index.
    """
    def __inc__(self, key, value, *args, **kwargs):
        if key in ('n_sub_batch', 'e_sub_batch'):
            return 1 + getattr(self, 'n_sub_batch')[-1]
        elif key == 'n_id':
            return self.num_nodes
        elif key == 'e_id':
            return self.edge_index.size(-1)
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def map_edge_index(self) -> torch.Tensor:
        """
        Map edge_index to the subgraph combined edge_index
        """
        edge_index = self.edge_index[:, self.e_id]
        n_nodes = self.x.size(0)
        n_sub_nodes = self.n_id.size(0)

        node_map = torch.ones((n_nodes, n_nodes)).long() * -1
        node_map[self.n_sub_batch, self.n_id] = torch.arange(n_sub_nodes)

        edge_index += (torch.arange(n_sub_nodes) * n_nodes)[self.e_sub_batch]
        print(node_map.shape)
        return node_map.reshape(-1)[edge_index]

    def map_data(self) -> Data:
        """
        Returns a :obj:`Data` object of the subgraph mapped data. Keeps the
        additional batch information for later aggregation.
        """
        data = Data()
        for k, v in self.items():
            if k in ('n_sub_batch', 'n_sub_batch', 'n_id', 'e_id'):
                data[k] = v
            elif k == 'edge_index':
                data[k] = self.map_edge_index()
            elif self.is_node_attr(k):
                data[k] = torch.index_select(v, self.__cat_dim__(k, v),
                                             self.n_id)
            elif self.is_edge_attr(k):
                data[k] = torch.index_select(v, self.__cat_dim__(k, v),
                                             self.e_id)
            else:
                data[k] = v
        return data


class RootedSubgraph(BaseTransform):
    r"""
    Base class for rooted subgraph.
    The object transforms a Data object to RootedSubgraphData object.
    """
    def extract(self, edge_index: torch.Tensor,
                num_nodes: int) -> RootedSubgraphData:
        r""" Create :obj:`RootedSubgraphData` object from an edge_index. The
        returned :obj:`RootedSubgraphData` does not contain the original graph
        data.
        """
        raise NotImplementedError

    def __call__(self, data: Data) -> Data:
        data = self.extract(data.edge_index, data.num_nodes)
        for k, v in data.to_dict():
            data[k] = v
        return data


class RootedEgoNets(RootedSubgraph):
    """ Record rooted k-hop Egonet for each node in the graph.
    From the `"From Stars to Subgraphs: Uplifting Any GNN with Local Structure
    Awareness" <https://arxiv.org/pdf/2110.03753.pdf>`_ paper

    Args:
        num_hops (int): k for k-hop Egonet.
    """
    def __init__(self, num_hops: int):
        super().__init__()
        self.num_hops = num_hops

    def extract(self, edge_index: torch.Tensor,
                num_nodes: int) -> RootedSubgraphData:
        # return k-hop subgraph for all nodes in the graph
        row, col = edge_index
        sparse_adj = SparseTensor(row=row, col=col,
                                  sparse_sizes=(num_nodes, num_nodes))
        hop_mask = sparse_adj.to_dense() > 0
        n_hops = torch.eye(num_nodes, dtype=torch.int,
                           device=edge_index.device) - 1
        for i in range(self.num_hops):
            n_hops[(n_hops == -1) & hop_mask] = i + 1
            hop_mask = sparse_adj.matmul(hop_mask.float()) > 0

        node_mask = (n_hops >= 0)  # N x N dense mask matrix
        return rooted_subgraph_from_mask(node_mask, edge_index, n_hops)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.num_hops})'


class RootedRWSubgraph(RootedSubgraph):
    """ Record rooted random-walk based subgraph for each node in the graph.
    From the `"From Stars to Subgraphs: Uplifting Any GNN with Local Structure
    Awareness" <https://arxiv.org/pdf/2110.03753.pdf>`_ paper

    Args:
        walk_length (int, optional): the length of random walk. When it is 0
            use k-hop Egonet.
        p (float, optional): parameters of node2vec's random walk.
        q (float, optional): parameters of node2vec's random walk.
        repeat (int, optional): times of repeating the random walk to reduce
            randomness.
        return_hops (bool, optional): whether return distance to centroid
        feature.
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

    def extract(self, edge_index: torch.Tensor,
                num_nodes: int) -> RootedSubgraphData:
        from torch_cluster import random_walk

        row, col = edge_index
        start = torch.arange(num_nodes, device=row.device)
        walks = [
            random_walk(row, col, start=start, walk_length=self.walk_length,
                        p=self.p, q=self.q, num_nodes=num_nodes)
            for _ in range(self.repeat)
        ]
        walk = torch.cat(walks, dim=-1)
        node_mask = row.new_empty((num_nodes, num_nodes), dtype=torch.bool)
        node_mask.fill_(False)
        index = start.repeat_interleave(
            (self.walk_length + 1) * self.repeat), walk.reshape(-1)
        node_mask[index] = True

        n_hops = None
        if self.return_hops:  # this is fast enough
            sparse_adj = SparseTensor(row=row, col=col,
                                      sparse_sizes=(num_nodes, num_nodes))
            hop_mask = sparse_adj.to_dense() > 0
            n_hops = torch.eye(num_nodes, dtype=torch.int,
                               device=edge_index.device) - 1

            for i in range(self.max_hops):
                n_hops[(n_hops == -1) & hop_mask] = i + 1
                hop_mask = sparse_adj.matmul(hop_mask.float()) > 0
                if n_hops[node_mask].min() != -1:
                    break
        return rooted_subgraph_from_mask(node_mask, edge_index, n_hops)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(l{self.walk_length}-' \
               f'p{self.p}-q{self.q}-r{self.repeat})'


def rooted_subgraph_from_mask(node_mask: torch.Tensor,
                              edge_index: torch.Tensor,
                              hop_mat: torch.Tensor) -> RootedSubgraphData:
    n_hops = (hop_mat + 1).nonzero()[:, 1] - 1
    n_sub_batch, n_id = node_mask.nonzero().T
    e_sub_batch, e_id = (node_mask[:, edge_index[0]]
                         & node_mask[:, edge_index[1]]).nonzero().T

    data = RootedSubgraphData()
    data.n_sub_batch = n_sub_batch
    data.n_id = n_id
    data.e_sub_batch = e_sub_batch
    data.e_id = e_id
    data.n_hops = n_hops
    return data
