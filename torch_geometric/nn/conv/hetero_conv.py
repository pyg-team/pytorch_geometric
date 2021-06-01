import torch
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
import torch.nn as nn

from torch import Tensor
from torch._six import container_abcs
from torch_geometric.nn.inits import reset
from typing import (
    List,
    Dict,
)


class HeteroConv(torch.nn.Module):
    r"""A "wrapper" layer designed for heterogeneous graph layers. It takes a
    heterogeneous graph layer, such as :class:`torch_geometric.conv.hetero_sage_conv.HeteroSAGEConv`, 
    at the initializing stage. 
    .. note::
        For more detailed use of :class:`HeteroConv`, TODO: example
    """
    def __init__(self, convs, aggr="add"):
        super(HeteroConv, self).__init__()

        assert isinstance(convs, container_abcs.Mapping)
        self.convs = convs
        self.modules = torch.nn.ModuleList(convs.values())

        assert aggr in ["add", "mean", "max", "mul", "concat", None]
        self.aggr = aggr

    def reset_parameters(self):
        r"""
        """
        for conv in self.convs.values():
            reset(conv)

    def forward(self, node_features, edge_indices, edge_features=None):
        r"""The forward function for :class:`HeteroConv`.
        Args:
            node_features (Dict[str, Tensor]): A dictionary each key is node type and the corresponding
                value is a node feature tensor.
            edge_indices (Dict[str, Tensor]): A dictionary each key is message type and the corresponding
                value is an `edge _ndex` tensor.
            edge_features (Dict[str, Tensor]): A dictionary each key is edge type and the corresponding
                value is an edge feature tensor. The default value is `None`.
        """
        
        # node embedding computed from each message type
        message_type_emb = {}
        for message_key, message_type in edge_indices.items():
            if message_key not in self.convs:
                continue
            neigh_type, edge_type, self_type = message_key
            node_feature_neigh = node_features[neigh_type]
            node_feature_self = node_features[self_type]
            # TODO: edge_features is not used
            if edge_features is not None:
                edge_feature = edge_features[edge_type]
            edge_index = edge_indices[message_key]

            message_type_emb[message_key] = self.convs[message_key](
                node_feature_neigh,
                node_feature_self,
                edge_index,
            )

        # aggregate node embeddings from different message types into 1 node
        # embedding for each node
        node_emb = {tail: [] for _, _, tail in message_type_emb.keys()}

        for (_, _, tail), item in message_type_emb.items():
            node_emb[tail].append(item)

        # Aggregate multiple embeddings with the same tail.
        for node_type, embs in node_emb.items():
            if len(embs) == 1:
                node_emb[node_type] = embs[0]
            else:
                node_emb[node_type] = self.aggregate(embs)

        return node_emb

    def aggregate(self, xs: List[Tensor]):
        r"""The aggregation for each node type. Currently support `concat`, `add`,
        `mean`, `max` and `mul`.
        Args:
            xs (List[Tensor]): A list of :class:`torch.Tensor` for a node type. 
                The number of elements in the list equals to the number of 
                `message types` that the destination node type is current node type.
        """
        if self.aggr == "concat":
            return torch.cat(xs, dim=-1)

        x = torch.stack(xs, dim=-1)
        if self.aggr == "add":
            return x.sum(dim=-1)
        elif self.aggr == "mean":
            return x.mean(dim=-1)
        elif self.aggr == "max":
            return x.max(dim=-1)[0]
        elif self.aggr == "mul":
            return x.prod(dim=-1)[0]


def forward_op(x, module_dict, **kwargs):
    r"""A helper function for the heterogeneous operations. Given a dictionary input
    `x`, it will return a dictionary with the same keys and the values applied by the
    corresponding values of the `module_dict` with specified parameters. The keys in `x` 
    are same with the keys in the `module_dict`.
    Args:
        x (Dict[str, Tensor]): A dictionary that the value of each item is a tensor.
        module_dict (:class:`torch.nn.ModuleDict`): The value of the `module_dict` 
            will be fed with each value in `x`.
        **kwargs (optional): Parameters that will be passed into each value of the 
            `module_dict`.
    """
    if not isinstance(x, dict):
        raise ValueError("The input x should be a dictionary.")
    res = {}
    for key in x:
        res[key] = module_dict[key](x[key], **kwargs)
    return res


def loss_op(pred, y, index, loss_func):
    r"""
    A helper function for the heterogeneous loss operations.
    This function will sum the loss of all node types.
    Args:
        pred (Dict[str, Tensor]): A dictionary of prediction results.
        y (Dict[str, Tensor]): A dictionary of labels. The keys should match with 
            the keys in the `pred`.
        index (Dict[str, Tensor]): A dictionary of indicies that the loss 
            will be computed on. Each value should be :class:`torch.LongTensor`. 
            Notice that `y` will not be indexed by the `index`. Here we assume 
            `y` has been splitted into proper sets.
        loss_func (callable): The defined loss function.
    """
    loss = 0
    for node_type in pred:
        idx = index[node_type]
        loss += loss_func(pred[node_type][idx], y[node_type])
    return loss
