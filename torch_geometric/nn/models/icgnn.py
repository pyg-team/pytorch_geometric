import math
import torch
from torch import Tensor
from torch.nn import Parameter, Module, init, ModuleList, Linear, Dropout
from typing import Optional, Dict, Any, Iterator

from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.nn.conv.simple_conv import MessagePassing


class FrobeniusNormMessagePassing(MessagePassing):
    def __init__(
        self,
        community_scalars: Tensor,
    ):
        super().__init__(aggr='sum')
        self.community_scalars = community_scalars  # (K,)

    def message(self, x_i: Tensor, x_j: Tensor, edge_weight: OptTensor = None) -> Tensor:
        # x_i is the target node of dim (E, K)
        # x_j is the source node of dim (E, K)
        message_term = (x_i * self.community_scalars * x_j).sum(dim=1, keepdims=True)  # (E, )
        if edge_weight is not None:
            message_term = message_term * edge_weight.unsqueeze(dim=1)
        return message_term

    def forward(self, affiliate_mat: Tensor, edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:
        node_based_result = self.propagate(edge_index, x=affiliate_mat, edge_weight=edge_weight)  # (N, 1)
        return node_based_result.squeeze(dim=1).sum(dim=0)  # (,)


class ICGFitter(Module):
    def __init__(
        self,
        num_nodes: int,
        K: int,
        feature_scale: float,
        in_channel: int,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.K = K
        self.in_channel = in_channel
        self.feature_scale = feature_scale

        # learned icg parameters
        self._affiliate_mat = Parameter(torch.zeros((num_nodes, K)))
        self.community_scalars = Parameter(torch.zeros((K,)))
        self.feat_mat = Parameter(torch.zeros((K, in_channel)))

    def reset_parameters(self):
        # feat_mat is inspired by torch.nn.modules.linear class Linear (as both are linear transformations)
        bound = 1 / math.sqrt(self.K)
        init.uniform_(self.feat_mat, -bound, bound)

        # community_scalars is inspired by the bias of torch.nn.modules.linear class Linear
        init.uniform_(self.community_scalars, -bound, bound)

        # _affiliate_mat
        init.uniform_(self._affiliate_mat, -4, 4)

        # feat_mat is inspired by torch.nn.modules.linear class Linear (as both are linear transformations)
        bound = 1 / math.sqrt(self.K)
        init.uniform_(self.feat_mat, -bound, bound)

    @property
    def affiliate_mat(self) -> Tensor:
        return torch.sigmoid(self._affiliate_mat)

    @property
    def affiliate_times_scalar(self) -> Tensor:
        return self.affiliate_mat * self.community_scalars

    def icg_fit_loss(self, x: Tensor, edge_index: Adj) -> Tensor:
        self.train()

        num_nodes, num_feat = x.shape[0], x.shape[1]
        frobenius_norm_mp = FrobeniusNormMessagePassing(community_scalars=self.community_scalars).to(device=x.device)

        # loss terms
        global_term = torch.trace(self.affiliate_mat.T @ self.affiliate_times_scalar @ self.affiliate_mat.T \
                                  @ (self.affiliate_mat * self.community_scalars))  # (,)
        local_term = 2 * frobenius_norm_mp(affiliate_mat=self.affiliate_mat, edge_index=edge_index)  # (,)
        loss = (global_term - local_term + edge_index.shape[1]) / num_nodes

        if self.feature_scale > 0:
            feature_loss = torch.sum((x - (self.affiliate_mat * self.community_scalars) @ self.feat_mat) ** 2) / num_feat
            loss = loss + self.feature_scale * feature_loss
        self.eval()
        return loss


class ICGNNLayer(Module):
    def __init__(
            self,
            K: int,
            in_channel: int,
            out_channel: int,
    ):
        super().__init__()

        self.community_mat = Parameter(torch.zeros((K, out_channel)))
        self.lin_node = Linear(in_channel, out_channel)
        self.lin_community = Linear(out_channel, out_channel)

    def forward(self, x: Tensor, affiliate_times_scalar: Tensor) -> Tensor:
        out = self.lin_community(affiliate_times_scalar @ self.community_mat)  # (N, C)
        out = out + self.lin_node(x)
        return out  # (N, C)


class ICGNNSequence(Module):
    def __init__(
            self,
            K: int,
            in_channel: int,
            hidden_channel: int,
            out_channel: int,
            num_layers: int,
            dropout: float,
            activation: str = 'relu',
            activation_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        dim_list = [in_channel] + [hidden_channel] * (num_layers - 1) + [out_channel]
        self.layers = ModuleList([ICGNNLayer(K=K, in_channel=in_channels, out_channel=out_channels)
                                  for in_channels, out_channels in zip(dim_list[:-1], dim_list[1:])])
        self.dropout = Dropout(dropout)
        self.activation = activation_resolver(activation, **(activation_kwargs or {}))

    def forward(self, x: Tensor, affiliate_times_scalar: Tensor) -> Tensor:
        for layer in self.layers[:-1]:
            x = layer(x=x, affiliate_times_scalar=affiliate_times_scalar)  # (N, C)
            x = self.activation(x)
            x = self.dropout(x)
        x = self.layers[-1](x=x, affiliate_times_scalar=affiliate_times_scalar)
        return x


class ICGNN(Module):
    r"""The ICGNN model from the `"Learning on Large Graphs using Intersecting Communities"
    <https://arxiv.org/abs/2405.20724>`_ paper.

    Args:
        num_nodes (int): The number of nodes in the graph.
        K (int): The number of communities.
        in_channel (int): The number of node input features.
        hidden_channel (int): The number of hidden features.
        out_channel (int): The number of output features.
        num_layers (int): The number of layers in the network.
        dropout (float): The dropout rate applied during training (default: :obj:`0.0`).
        feature_scale (float, optional): A scaling factor applied to the node features in the
            ICG approximation loss (default: :obj:`0.0`).
        activation (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        activation_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`activation`.
            (default: :obj:`None`)
    """
    def __init__(
        self,
        num_nodes: int,
        K: int,
        in_channel: int,
        hidden_channel: int,
        out_channel: int,
        num_layers: int,
        dropout: float,
        feature_scale: Optional[float] = 0,
        activation: str = 'relu',
        activation_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self.icg_fitter = ICGFitter(
            num_nodes=num_nodes,
            K=K,
            feature_scale=feature_scale,
            in_channel=in_channel,
        )

        self.nn = ICGNNSequence(
            K=K,
            in_channel=in_channel,
            hidden_channel=hidden_channel,
            out_channel=out_channel,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            activation_kwargs=activation_kwargs,
        )

    # gradients
    def train(self, mode: bool = True):
        self.icg_fitter.eval()
        self.nn.train()

    def parameters(self, recurse: bool = True):
        return NotImplementedError('Use nn_parameters or icg_parameters')

    def icg_parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return self.icg_fitter.parameters()

    def nn_parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return self.nn.parameters()

    def forward(self, x: Tensor) -> Tensor:
        return self.nn(
            x=x,
            affiliate_times_scalar=self.icg_fitter.affiliate_times_scalar,
        )

    def icg_fit_loss(self, x: Tensor, edge_index: Adj) -> Tensor:
        return self.icg_fitter.icg_fit_loss(x=x, edge_index=edge_index)
