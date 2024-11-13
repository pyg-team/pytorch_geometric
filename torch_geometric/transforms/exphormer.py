import torch.nn as nn

from torch_geometric.nn.attention.expander import ExpanderAttention
from torch_geometric.nn.attention.local import LocalAttention
from torch_geometric.transforms import VirtualNode


class EXPHORMER(nn.Module):
    """EXPHORMER architecture.
    Based on the paper: https://arxiv.org/abs/2303.06147
    """
    def __init__(self, hidden_dim: int, num_layers: int = 3,
                 num_heads: int = 4, expander_degree: int = 4,
                 dropout: float = 0.1, use_expander: bool = True,
                 use_global: bool = True, num_virtual_nodes: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_expander = use_expander
        self.use_global = use_global
        self.virtual_node_transform = VirtualNode() if use_global else None
        if use_global and num_virtual_nodes < 1:
            raise ValueError(
                "num_virtual_nodes must be at least 1 if use_global is enabled."
            )
        self.num_virtual_nodes = num_virtual_nodes
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'local':
                LocalAttention(hidden_dim, num_heads=num_heads,
                               dropout=dropout),
                'expander':
                ExpanderAttention(hidden_dim, expander_degree=expander_degree,
                                  num_heads=num_heads, dropout=dropout)
                if use_expander else None,
                'layer_norm':
                nn.LayerNorm(hidden_dim),
                'ffn':
                nn.Sequential(nn.Linear(hidden_dim, 4 * hidden_dim), nn.GELU(),
                              nn.Dropout(dropout),
                              nn.Linear(4 * hidden_dim, hidden_dim))
            }) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        if data.x.size(0) == 0:
            raise ValueError("Input graph is empty.")
        if not hasattr(data, 'edge_index') or data.edge_index is None:
            raise ValueError(
                "Input data must contain 'edge_index' for message passing.")
        x, edge_index = data.x, data.edge_index
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        batch_size = x.size(0)
        if self.virtual_node_transform is not None:
            data = self.virtual_node_transform(data)
            x, edge_index = data.x, data.edge_index

        for layer in self.layers:
            residual = x
            local_out = layer['local'](x, edge_index, edge_attr)
            expander_out = 0
            if self.use_expander and layer['expander'] is not None:
                expander_out, _ = layer['expander'](
                    x[:batch_size + self.num_virtual_nodes],
                    batch_size + self.num_virtual_nodes)
            x = layer['layer_norm'](residual + local_out + expander_out)
            x = x + layer['ffn'](x)
        return x[:batch_size]
