import torch

from torch_geometric.data import Data
from torch_geometric.nn.models import TensorGNAN


def _dummy_data(num_nodes: int = 5, num_feats: int = 4):
    x = torch.randn(num_nodes, num_feats)
    edge_index = torch.combinations(torch.arange(num_nodes), r=2).t()
    # full distance matrix:
    rand_dist = torch.rand(num_nodes, num_nodes)
    rand_dist = (rand_dist + rand_dist.t()) / 2  # symmetric
    norm = rand_dist.sum(dim=1)
    data = Data(x=x, edge_index=edge_index)
    data.node_distances = rand_dist
    data.normalization_matrix = norm
    return data


def test_tensor_gnan_graph_level():
    data = _dummy_data()
    model = TensorGNAN(in_channels=data.num_features, out_channels=2,
                       n_layers=2, hidden_channels=8)
    out = model(data)  # [1, 2]
    assert out.shape == (1, 2)
