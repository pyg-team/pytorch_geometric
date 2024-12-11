import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv


class NBNormZeroInflated(nn.Module):
    """A PyTorch Geometric model for a Zero-Inflated Negative Binomial (ZINB)
    distribution applied to graph-structured data. This model predicts three
    key parameters of the ZINB distribution for each node in the graph:
        - n: Count parameter (positive values).
        - p: Success probability (values between 0 and 1).
        - pi: Zero-inflation parameter (probability of structural zeros).

    Args:
        c_in (int): Number of input features for each node.
        c_out (int): Number of output features for each node (dimension of
            output parameters).

    Attributes:
        n_conv (GCNConv): Graph convolutional layer for computing the count
            parameter `n`.
        p_conv (GCNConv): Graph convolutional layer for computing the
            probability parameter `p`.
        pi_conv (GCNConv): Graph convolutional layer for computing the
            zero-inflation parameter `pi`.
        out_dim (int): Output dimension (same as `c_out`).
    """
    def __init__(self, c_in, c_out):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out

        # Graph convolutional layers
        self.n_conv = GCNConv(c_in, c_out)  # For count parameter
        self.p_conv = GCNConv(c_in, c_out)  # For success probability
        self.pi_conv = GCNConv(c_in, c_out)  # For zero-inflation probability

        self.out_dim = c_out  # Output dimension

    def forward(self, x, edge_index):
        """Forward pass through the model.

        Args:
            x (torch.Tensor): Node feature matrix of shape [num_nodes, c_in].
                Each row corresponds to the features of a single node.
            edge_index (torch.Tensor): Edge list in COO format of shape
                [2, num_edges]. Defines the graph's connectivity.

        Returns:
            tuple:
                - n (torch.Tensor): Predicted count parameter for each node,
                    shape [num_nodes, c_out]. Positive values ensured by
                    applying the `softplus` function.
                - p (torch.Tensor): Predicted success probability for each
                    node, shape [num_nodes, c_out]. Values in range (0, 1)
                    ensured by applying the `sigmoid` function.
                - pi (torch.Tensor): Predicted zero-inflation probability for
                    each node, shape [num_nodes, c_out]. Values in range (0, 1)
                    ensured by applying the `sigmoid` function.
        """
        # Graph convolutional layers
        n = self.n_conv(x, edge_index)  # Output count parameter
        p = self.p_conv(x, edge_index)  # Output success probability
        pi = self.pi_conv(x, edge_index)  # Output zero-inflation probability

        # Ensure outputs are in valid ranges
        n = F.softplus(n)  # Ensures positivity
        p = torch.sigmoid(p)  # Ensures values are between 0 and 1
        pi = torch.sigmoid(pi)  # Ensures values are between 0 and 1

        return n, p, pi
