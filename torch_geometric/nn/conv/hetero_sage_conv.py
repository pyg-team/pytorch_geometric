import torch
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
import torch.nn as nn

from torch import Tensor
from torch._six import container_abcs
from torch_sparse import matmul
from typing import (
    List,
    Dict,
)


class HeteroSAGEConv(pyg_nn.MessagePassing):
    r"""The heterogeneous compitable GraphSAGE operator is derived from the `"Inductive Representation
    Learning on Large Graphs" <https://arxiv.org/abs/1706.02216>`_, `"Modeling polypharmacy side
    effects with graph convolutional networks" <https://arxiv.org/abs/1802.00543>`_, and `"Modeling
    Relational Data with Graph Convolutional Networks" <https://arxiv.org/abs/1703.06103>`_ papers.
    .. note::
        This layer is a component to be used with the :class:`HeteroConv`.
        The Message passing for the message type (node_type1, edge_type, node_type2).
    Args:
        in_channels_neighbor (int): The input dimension of the neighbor node type.
            Corresponds to the input dimension of node_type1.
        out_channels (int): The dimension of the output.
        in_channels_self (int): The input dimension of the self node type.
            Default is `None` where the `in_channels_self` is equal to `in_channels_neigh`.
        remove_self_loop (bool): Whether to remove self loops using :class:`torch_geometric.utils.remove_self_loops`.
            Default is `True`.
    """
    def __init__(self, in_channels_neighbor, out_channels,
                 in_channels_self=None, remove_self_loop=True):
        super(HeteroSAGEConv, self).__init__(aggr="add")
        self.remove_self_loop = remove_self_loop
        self.in_channels_neigh = in_channels_neighbor
        if in_channels_self is None:
            self.in_channels_self = in_channels_neighbor
        else:
            self.in_channels_self = in_channels_self
        self.out_channels = out_channels
        self.lin_neigh = nn.Linear(self.in_channels_neigh, self.out_channels)
        self.lin_self = nn.Linear(self.in_channels_self, self.out_channels)
        self.lin_update = nn.Linear(self.out_channels * 2, self.out_channels)

    def forward(
        self,
        node_feature_neigh,
        node_feature_self,
        edge_index,
        edge_weight=None,
        size=None,
        res_n_id=None,
    ):
        r"""
        """
        if self.remove_self_loop:
            edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        return self.propagate(edge_index, size=size,
                              node_feature_neigh=node_feature_neigh,
                              node_feature_self=node_feature_self,
                              edge_weight=edge_weight, res_n_id=res_n_id)

    def message(self, node_feature_neigh_j, node_feature_self_i, edge_weight):
        r"""
        """
        return node_feature_neigh_j

    def message_and_aggregate(self, edge_index, node_feature_neigh):
        r"""
        This function basically fuses the :meth:`message` and :meth:`aggregate` into 
        one function. It will save memory and avoid message materialization. More 
        information please refer to the PyTorch Geometric documentation.
        Args:
            edge_index (:class:`torch_sparse.SparseTensor`): The `edge_index` sparse tensor.
            node_feature_neigh (:class:`torch.Tensor`): Neighbor feature tensor.
        """
        out = matmul(edge_index, node_feature_neigh, reduce="mean")
        return out

    def update(self, aggr_out, node_feature_self, res_n_id):
        r"""
        """
        aggr_out = self.lin_neigh(aggr_out)
        node_feature_self = self.lin_self(node_feature_self)
        aggr_out = torch.cat([aggr_out, node_feature_self], dim=-1)
        aggr_out = self.lin_update(aggr_out)
        return aggr_out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"(neigh: {self.in_channels_neigh}, self: {self.in_channels_self}, "
            f"out: {self.out_channels})")
