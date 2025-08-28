import torch.nn as nn

# Use alias import to avoid yapf/isort conflicts with long module names
import torch_geometric.contrib.nn.layers.activation as _activation
from torch_geometric.nn.inits import reset

get_activation_function = _activation.get_activation_function


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

    def reset_parameters(self) -> None:
        """Reset parameters of the feedforward layer.

        This method should reset all learnable parameters of the module.
        It is called during initialization and can be overridden by subclasses.

        Args:
            None
        Returns:
            None
        """
        for module in [self.fc1, self.fc2, self.ffn_layer_norm]:
            reset(module)
