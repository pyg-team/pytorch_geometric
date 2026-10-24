import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing


# Spline-related utility functions
def B_batch(x, grid, k=0):
    """Evaluate x on B-spline bases.
    """
    x = x.unsqueeze(dim=2)
    grid = grid.unsqueeze(dim=0)

    if k == 0:
        value = (x >= grid[:, :, :-1]) * (x < grid[:, :, 1:])
    else:
        B_km1 = B_batch(x[:, :, 0], grid=grid[0], k=k - 1)
        value = (
            (x - grid[:, :, :-(k + 1)]) /
            (grid[:, :, k:-1] - grid[:, :, :-(k + 1)]) * B_km1[:, :, :-1] +
            (grid[:, :, k + 1:] - x) /
            (grid[:, :, k + 1:] - grid[:, :, 1:(-k)]) * B_km1[:, :, 1:])

    # Handle degenerate cases
    value = torch.nan_to_num(value)
    return value


def coef2curve(x_eval, grid, coef, k):
    """Convert B-spline coefficients to B-spline curves and evaluate x on them.
    """
    b_splines = B_batch(x_eval, grid, k=k)
    y_eval = torch.einsum('ijk,jlk->ijl', b_splines, coef.to(b_splines.device))
    return y_eval


def curve2coef(x_eval, y_eval, grid, k):
    """Convert B-spline curves to B-spline coefficients using least squares.
    """
    batch = x_eval.shape[0]
    in_dim = x_eval.shape[1]
    out_dim = y_eval.shape[2]
    n_coef = grid.shape[1] - k - 1
    mat = B_batch(x_eval, grid, k)
    mat = mat.permute(1, 0, 2)[:, None, :, :].expand(in_dim, out_dim, batch,
                                                     n_coef)
    y_eval = y_eval.permute(1, 2, 0).unsqueeze(dim=3)

    # Solve least squares
    coef = torch.linalg.lstsq(mat, y_eval).solution[:, :, :, 0]
    return coef


def extend_grid(grid, k_extend=0):
    """Extend the grid by k points on both sides for smoother splines.
    """
    h = (grid[:, [-1]] - grid[:, [0]]) / (grid.shape[1] - 1)

    for i in range(k_extend):
        grid = torch.cat([grid[:, [0]] - h, grid], dim=1)
        grid = torch.cat([grid, grid[:, [-1]] + h], dim=1)

    return grid


# KANLayer
class KANLayer(nn.Module):
    """Implements a spline-based activation layer for GKAN.
    """
    def __init__(self, in_dim, out_dim, num=5, k=3, noise_scale=0.5,
                 grid_range=[-1, 1], base_fun=torch.nn.SiLU()):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num = num
        self.k = k
        self.base_fun = base_fun

        # Initialize grid with k-extended values
        grid = torch.linspace(grid_range[0], grid_range[1],
                              steps=num + 1)[None, :].expand(in_dim, num + 1)
        extended_grid = extend_grid(grid, k_extend=k)
        self.register_buffer('grid', extended_grid)  # Non-trainable grid

        # Initialize spline coefficients with noise
        noises = (torch.rand(num + 1, in_dim, out_dim) -
                  0.5) * noise_scale / num
        self.coef = nn.Parameter(
            curve2coef(self.grid[:, k:-k].permute(1, 0), noises, self.grid, k))

        # Trainable scaling parameters
        self.scale_base = nn.Parameter(torch.ones(in_dim, out_dim))
        self.scale_sp = nn.Parameter(torch.ones(in_dim, out_dim))

    def forward(self, x):
        """Forward pass through the KANLayer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_dim).
        """
        base = self.base_fun(x)  # Residual function b(x)
        y = coef2curve(x, grid=self.grid, coef=self.coef,
                       k=self.k)  # Spline function
        y = self.scale_base[None, :, :] * base[:, :, None] + self.scale_sp[
            None, :, :] * y
        return torch.sum(y, dim=1)


# GKAN Layer
class GKAN(MessagePassing):
    """Graph Kolmogorov-Arnold Network (GKAN) layer.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num=5, k=3,
                 architecture=2):
        super().__init__(aggr="add")
        self.architecture = architecture

        # KAN Layer (spline-based activation)
        self.kan_layer = KANLayer(input_dim, hidden_dim, num=num, k=k)

        # Linear transformation for output
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        """Forward pass for GKAN layer.
        """
        if self.architecture == 1:
            # Aggregate first, then apply KAN
            x = self.propagate(edge_index, x=x)
            x = self.kan_layer(x)
        elif self.architecture == 2:
            # Apply KAN first, then aggregate
            x = self.kan_layer(x)
            x = self.propagate(edge_index, x=x)
        return self.linear(x)

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return aggr_out


# GKAN Model
class GKANModel(nn.Module):
    """Graph Kolmogorov-Arnold Network (GKAN) model for node classification.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, num=5,
                 k=3, architecture=2):
        super().__init__()
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(
            GKAN(input_dim, hidden_dim, hidden_dim, num=num, k=k,
                 architecture=architecture))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(
                GKAN(hidden_dim, hidden_dim, hidden_dim, num=num, k=k,
                     architecture=architecture))

        # Output layer
        self.layers.append(
            GKAN(hidden_dim, hidden_dim, output_dim, num=num, k=k,
                 architecture=architecture))

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = F.relu(layer(x, edge_index))
        return x
