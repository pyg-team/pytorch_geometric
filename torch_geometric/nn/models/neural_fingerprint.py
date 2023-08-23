import torch
from torch import tensor
import torch.nn.functional as F

from torch_geometric.nn.conv import MFConv
from torch_geometric.nn.dense import Linear

class NeuralFingerprint(torch.nn.Module):
    def __init__(self, num_features, fingerprint_length, num_layers):
        super().__init__()
        self.num_features = num_features
        self.fingerprint_length = fingerprint_length
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(MFConv(in_channels=self.num_features, out_channels=self.num_features))
            self.layers.append(Linear(in_channels=self.num_features, out_channels=self.fingerprint_length))
    
    def forward(self, x, edge_index):
        fingerprint = torch.zeros(self.fingerprint_length)
        for i in range(0,2*self.num_layers,2):
            x = self.layers[i](x, edge_index)
            x = torch.sigmoid(x)
            y = F.softmax(self.layers[i+1](x), dim=1)
            for j in range(y.shape[0]):
                fingerprint += y[j]
        return fingerprint
    