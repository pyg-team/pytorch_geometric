import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionGCN(nn.Module):
    """A Diffusion Graph Convolutional Network (GCN) that applies
    a diffusion-based graph convolution to sampled locations.
    This implementation is adapted for PyTorch Geometric.

    Args:
        in_channels (int): Number of input features per node.
        out_channels (int): Desired number of output features per node.
        orders (int): Number of diffusion steps.
        activation (str, optional): Activation function to use ('relu' or
            'selu'). Default is 'relu'.

    Attributes:
        orders (int): The number of diffusion steps.
        activation (str): The activation function used.
        num_matrices (int): The total number of diffusion matrices.
        Theta1 (nn.Parameter): Weight matrix for graph convolution.
        bias (nn.Parameter): Bias term added to the output.
    """
    def __init__(self, in_channels, out_channels, orders, activation='relu'):
        super().__init__()
        self.orders = orders
        self.activation = activation
        self.num_matrices = 2 * self.orders + 1

        # Define trainable parameters
        self.Theta1 = nn.Parameter(
            torch.FloatTensor(in_channels * self.num_matrices, out_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """Initializes the model parameters with a uniform distribution
        based on the number of output features.
        """
        stdv = 1.0 / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)
        stdv1 = 1.0 / math.sqrt(self.bias.shape[0])
        self.bias.data.uniform_(-stdv1, stdv1)

    def _concat(self, x, x_):
        """Concatenates the input tensor with the new tensor along the first
        dimension.

        Args:
            x (torch.Tensor): Existing tensor of shape [..., ...].
            x_ (torch.Tensor): Tensor to concatenate, of shape [..., ...].

        Returns:
            torch.Tensor: Concatenated tensor along dimension 0.
        """
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)

    def forward(self, X, A_q, A_h):
        """Forward pass of the Diffusion GCN.

        Args:
            X (torch.Tensor): Input data of shape (batch_size, num_nodes,
                num_timesteps).
            A_q (torch.Tensor): Forward random walk matrix of shape
                (num_nodes, num_nodes).
            A_h (torch.Tensor): Backward random walk matrix of shape
                (num_nodes, num_nodes).

        Returns:
            torch.Tensor: Output data of shape (batch_size, num_nodes,
                num_features).
        """
        batch_size = X.shape[0]
        num_nodes = X.shape[1]
        input_size = X.size(2)

        # Collect adjacency matrices
        supports = [A_q, A_h]

        # Reshape input for diffusion operation
        x0 = X.permute(1, 2, 0)  # (num_nodes, num_times, batch_size)
        x0 = x0.reshape(num_nodes, input_size * batch_size)
        x = x0.unsqueeze(0)

        # Diffusion steps
        for support in supports:
            x1 = torch.mm(support, x0)
            x = self._concat(x, x1)
            for k in range(2, self.orders + 1):
                x2 = 2 * torch.mm(support, x1) - x0
                x = self._concat(x, x2)
                x1, x0 = x2, x1

        # Reshape and apply weights
        x = x.reshape(self.num_matrices, num_nodes, input_size,
                      batch_size).permute(
                          3, 1, 2,
                          0)  # (batch_size, num_nodes, input_size, order)

        x = x.reshape(batch_size, num_nodes, input_size * self.num_matrices)

        # Ensure dimensions match
        if x.size(2) != self.Theta1.size(0):
            raise ValueError(f"Shape mismatch: x has {x.size(1)} columns, "
                             f"but Theta1 has {self.Theta1.size(0)} rows.")

        x = torch.matmul(x, self.Theta1)  # Apply weights
        x += self.bias  # Add bias

        # Apply activation
        if self.activation == 'relu':
            x = F.relu(x)
        elif self.activation == 'selu':
            x = F.selu(x)

        return x
