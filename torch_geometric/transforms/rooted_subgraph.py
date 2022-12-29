import copy
from abc import ABC, abstractmethod
from typing import Any, Tuple

import torch
from torch import Tensor

from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class RootedSubgraphData(Data):
    r"""A data object describing a homogeneous graph together with each node's
    rooted subgraph. It contains several additional properties that hold the
    information to map to batch of every node's rooted subgraph:

    * :obj:`sub_edge_index` (Tensor): The edge indices of all combined rooted
      subgraphs.
    * :obj:`n_id` (Tensor): The indices of nodes in all combined rooted
      subgraphs.
    * :obj:`e_id` (Tensor): The indices of edges in all combined rooted
      subgraphs.
    * :obj:`n_sub_batch` (Tensor): The batch vector to distinguish nodes across
      different subgraphs.
    * :obj:`e_sub_batch` (Tensor): The batch vector to distinguish edges across
      different subgraphs.
    """
    def __inc__(self, key, value, *args, **kwargs) -> Any:
        if key == 'sub_edge_index':
            return self.n_id.size(0)
        if key in ['n_sub_batch', 'e_sub_batch']:
            return 1 + int(self.n_sub_batch[-1])
        elif key == 'n_id':
            return self.num_nodes
        elif key == 'e_id':
            return self.edge_index.size(1)
        return super().__inc__(key, value, *args, **kwargs)

    def map_data(self) -> Data:
        # Maps all feature information of the :class:`Data` object to each
        # rooted subgraph.
        data = copy.copy(self)

        for key, value in self.items():
            if key in ['sub_edge_index', 'n_id', 'e_id', 'e_sub_batch']:
                del data[key]
            elif key == 'n_sub_batch':
                continue
            elif key == 'num_nodes':
                data.num_nodes = self.n_id.size(0)
            elif key == 'edge_index':
                data.edge_index = self.sub_edge_index
            elif self.is_node_attr(key):
                dim = self.__cat_dim__(key, value)
                data[key] = value.index_select(dim, self.n_id)
            elif self.is_edge_attr(key):
                dim = self.__cat_dim__(key, value)
                data[key] = value.index_select(dim, self.e_id)

        return data


class RootedSubgraph(BaseTransform, ABC):
    r"""Base class for implementing rooted subgraph transformations."""
    @abstractmethod
    def extract(
        self,
        data: Data,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        # Returns the tuple:
        # :obj:`(sub_edge_index, n_id, e_id, n_sub_batch, e_sub_batch)`
        # of the :class:`RootedSubgraphData` object.
        pass

    def map(
        self,
        data: Data,
        n_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:

        n_sub_batch, n_id = n_mask.nonzero().t()
        e_mask = n_mask[:, data.edge_index[0]] & n_mask[:, data.edge_index[1]]
        e_sub_batch, e_id = e_mask.nonzero().t()

        sub_edge_index = data.edge_index[:, e_id]
        arange = torch.arange(n_id.size(0), device=data.edge_index.device)
        node_map = data.edge_index.new_ones(data.num_nodes, data.num_nodes)
        node_map[n_sub_batch, n_id] = arange
        sub_edge_index += (arange * data.num_nodes)[e_sub_batch]
        sub_edge_index = node_map.view(-1)[sub_edge_index]

        return sub_edge_index, n_id, e_id, n_sub_batch, e_sub_batch

    def __call__(self, data: Data) -> RootedSubgraphData:
        out = self.extract(data)
        d = RootedSubgraphData.from_dict(data.to_dict())
        d.sub_edge_index, d.n_id, d.e_id, d.n_sub_batch, d.e_sub_batch = out
        return d


class RootedEgoNets(RootedSubgraph):
    r"""Collects rooted :math:`k`-hop EgoNets for each node in the graph, as
    described in the `"From Stars to Subgraphs: Uplifting Any GNN with Local
    Structure Awareness" <https://arxiv.org/abs/2110.03753>`_ paper.

    Args:
        num_hops (int): the number of hops :math:`k`.
    """
    def __init__(self, num_hops: int):
        super().__init__()
        self.num_hops = num_hops

    def extract(
        self,
        data: Data,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        from torch_sparse import SparseTensor

        adj_t = SparseTensor.from_edge_index(
            data.edge_index,
            sparse_sizes=(data.num_nodes, data.num_nodes),
        ).t()

        n_mask = torch.eye(data.num_nodes, device=data.edge_index.device)
        for _ in range(self.num_hops):
            n_mask += adj_t @ n_mask

        return self.map(data, n_mask > 0)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_hops={self.num_hops})'


class RootedRWSubgraph(RootedSubgraph):
    """Collects rooted random-walk based subgraphs for each node in the graph,
    as described in the `"From Stars to Subgraphs: Uplifting Any GNN with Local
    Structure Awareness" <https://arxiv.org/abs/2110.03753>`_ paper.

    Args:
        walk_length (int): the length of the random walk.
        repeat (int, optional): The number of times of repeating the random
            walk to reduce randomness. (default: :obj:`1`)
    """
    def __init__(self, walk_length: int, repeat: int = 1):
        super().__init__()
        self.walk_length = walk_length
        self.repeat = repeat

    def extract(
        self,
        data: Data,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        from torch_cluster import random_walk

        start = torch.arange(data.num_nodes, device=data.edge_index.device)
        start = start.view(-1, 1).repeat(1, self.repeat).view(-1)
        walk = random_walk(data.edge_index[0], data.edge_index[1], start,
                           self.walk_length, num_nodes=data.num_nodes)

        n_mask = torch.zeros((data.num_nodes, data.num_nodes),
                             dtype=torch.bool, device=walk.device)
        start = start.view(-1, 1).repeat(1, (self.walk_length + 1)).view(-1)
        n_mask[start, walk.view(-1)] = True

        return self.map(data, n_mask)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(walk_length={self.walk_length})'
