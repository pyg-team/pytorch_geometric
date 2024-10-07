from typing import Optional, Dict, Any
from torch.nn import Module, Dropout, ModuleList
from torch import Tensor
from torch_geometric.typing import Adj
import torch.nn.functional as F
from torch_geometric.nn.resolver import activation_resolver


class CoGNN(Module):
    r"""The CoGNN model from the `"Cooperative Graph Neural Netowrks"
    <https://arxiv.org/abs/2310.01267>`_ paper.
    
    Args:
        env_net (Module): The environment network.
        action_net (Module): The action network.
        env_activation (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        env_activation_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`activation`.
            (default: :obj:`None`)
        temp (float, Optional): The gumbel softmax temperature. (default: :obj:`0.01`)
        dropout (float, Optional): The dropout ratio. (default: :obj:`0.0`)

    .. note::
        The env_net is assumed to have the same input dimension and the action_net is assumed to have an output
        dimension of 4, see `examples/cognn.py <https://github.com/pyg-team/
        pytorch_geometric/blob/master/examples/cognn.py>`_ and `"Cooperative Graph Neural Netowrks"
        <https://arxiv.org/abs/2310.01267>`_ paper.
    """
    def __init__(
        self,
        env_net: ModuleList,
        action_net: Module,
        env_activation: str = 'relu',
        env_activation_kwargs: Optional[Dict[str, Any]] = None,
        temp: Optional[float] = 0.01,
        dropout: Optional[float] = 0.0,
    ):
        super().__init__()

        self.env_net = env_net
        self.action_net = action_net
        self.activation = activation_resolver(env_activation, **(env_activation_kwargs or {}))
        self.temp = temp
        self.dropout = Dropout(p=dropout)

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        u, v = edge_index

        for env_layer in self.env_net:
            action_logits = self.action_net(x=x, edge_index=edge_index)  # (N, 4)

            # sampling actions
            incoming_edge_prob = F.gumbel_softmax(logits=action_logits[:, :2], tau=self.temp, hard=True)
            outgoing_edge_prob = F.gumbel_softmax(logits=action_logits[:, 2:], tau=self.temp, hard=True)

            # creating subgraph
            keep_incoming_prob = incoming_edge_prob[:, 0]
            keep_outgoing_prob = outgoing_edge_prob[:, 0]
            edge_weight = keep_incoming_prob[v] * keep_outgoing_prob[u]

            # message propagation
            x = env_layer(x=x, edge_index=edge_index, edge_weight=edge_weight)
            x = self.dropout(x)
            x = self.activation(x)
        return x
