import torch
import torch.nn.functional as F

from torch_geometric.nn import Linear, MFConv
from torch_geometric.typing import Adj


class NeuralFingerprint(torch.nn.Module):
    r"""The Nueral Fingerprint model from the
    `"Convolutional Networks on Graphs for Learning Molecular Fingerprints"
    <https://arxiv.org/pdf/1509.09292.pdf>`__ paper to generate fingerprints
    of molecules.

    Args:
        num_features (int): The number of features of each node/atom.
        fingerprint_length (int): The length of fingerprint vector required.
        num_layers (int): Number of layers in the model (Radius in the paper).
    """
    def __init__(self, num_features: int, fingerprint_length: int,
                 num_layers: int):
        super().__init__()
        self.num_features = num_features
        self.fingerprint_length = fingerprint_length
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(
                MFConv(in_channels=self.num_features,
                       out_channels=self.num_features))
            self.layers.append(
                Linear(in_channels=self.num_features,
                       out_channels=self.fingerprint_length))

    def forward(self, x: torch.Tensor, edge_index: Adj) -> torch.Tensor:

        fingerprint = x.new_zeros(self.fingerprint_length)
        for i in range(0, 2 * self.num_layers, 2):
            x = self.layers[i](x, edge_index)
            x = torch.sigmoid(x)
            y = F.softmax(self.layers[i + 1](x), dim=1)
            fingerprint += torch.sum(y, dim=0)
        return fingerprint
