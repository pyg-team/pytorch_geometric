import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn import EGTConv
from torch_geometric.utils import to_dense_adj


class EGT(torch.nn.Module):
    r"""The Edge-augmented Graph Transformer (EGT) from the
    `"Global Self-Attention as a Replacement for Graph Convolution"
    <https://arxiv.org/abs/2108.03348>`_ paper.
    This model is built for node classification.

    Args:
        node_channels (int): Input channels.
        edge_channels (int): Edge channels.
        out_channels (int): Output channels.
        edge_update (bool, optional): Whether to update edge features.
            (default: :obj:`False`)
        num_layers (int, optional): The number of EGTConv layers.
            (default: :obj:`3`)
        num_heads (int, optional): The number of heads for attention.
            (default: :obj:`4`)
        dropout (float, optional): Dropout rate.
            (default: :obj:`0.3`)
    """
    def __init__(
        self,
        node_channels: int,
        edge_channels: int,
        out_channels: int,
        edge_update: bool = False,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.edge_update = edge_update
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                EGTConv(
                    channels=node_channels,
                    edge_dim=edge_channels,
                    edge_update=edge_update,
                    heads=num_heads,
                    attn_dropout=dropout,
                    num_virtual_nodes=0,
                ))

        self.fc = torch.nn.Linear(node_channels, out_channels)

    def reset_parameters(self) -> None:
        for conv in self.convs:
            conv.reset_parameters()
        self.fc.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
    ) -> Tensor:
        edge_attr = to_dense_adj(edge_index=edge_index, edge_attr=edge_attr)

        for conv in self.convs:
            if self.edge_update:
                x, edge_attr = conv(x, edge_attr)
            else:
                x = conv(x, edge_attr)

        x = self.fc(x)
        return F.log_softmax(x, dim=-1)
