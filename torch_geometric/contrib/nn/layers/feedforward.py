import torch.nn as nn

from torch_geometric.contrib.nn.layers.activation import (
    get_activation_function,
)


class PositionwiseFeedForward(nn.Module):
    def __init__(
        self,
        hidden_dim,
        ffn_hidden_dim,
        dropout,
        activation='relu',
    ):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, ffn_hidden_dim)
        self.fc2 = nn.Linear(ffn_hidden_dim, hidden_dim)
        self.ffn_act_func = nn.ReLU()
        self.act_dropout = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)
        self.ffn_layer_norm = nn.LayerNorm(hidden_dim)
        self.ffn_act_func = get_activation_function(activation)

    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.ffn_act_func(x)
        x = self.act_dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = x + residual
        x = self.ffn_layer_norm(x)
        return x
