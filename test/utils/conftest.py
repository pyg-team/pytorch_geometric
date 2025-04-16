import pytest
import torch

from torch_geometric.data import HeteroData
from torch_geometric.explain.config import ModelMode, ModelReturnType
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.testing import get_random_edge_index


class GraphSAGE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), 32)
        self.conv2 = SAGEConv((-1, -1), 32)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


class HeteroSAGE(torch.nn.Module):
    def __init__(self, metadata, model_config=None):
        super().__init__()
        self.model_config = model_config
        self.graph_sage = to_hetero(GraphSAGE(), metadata, debug=False)

        # Determine output channels based on model_config
        out_channels = 1
        if (model_config
                and model_config.mode == ModelMode.multiclass_classification):
            out_channels = 7

        self.lin = torch.nn.Linear(32, out_channels)

    def forward(self, x_dict, edge_index_dict,
                additonal_arg=None) -> torch.Tensor:
        x = self.lin(self.graph_sage(x_dict, edge_index_dict)['paper'])

        # Apply transformations based on model_config if available
        if hasattr(self, 'model_config') and self.model_config:
            if self.model_config.mode == ModelMode.binary_classification:
                if self.model_config.return_type == ModelReturnType.probs:
                    x = x.sigmoid()
            elif self.model_config.mode == ModelMode.multiclass_classification:
                if self.model_config.return_type == ModelReturnType.probs:
                    x = x.softmax(dim=-1)
                elif (self.model_config.return_type ==
                      ModelReturnType.log_probs):
                    x = x.log_softmax(dim=-1)

        return x


@pytest.fixture()
def hetero_data():
    data = HeteroData()
    data['paper'].x = torch.randn(8, 16)
    data['author'].x = torch.randn(10, 8)

    data['paper', 'paper'].edge_index = get_random_edge_index(8, 8, 10)
    data['paper', 'paper'].edge_attr = torch.randn(10, 16)
    data['paper', 'author'].edge_index = get_random_edge_index(8, 10, 10)
    data['paper', 'author'].edge_attr = torch.randn(10, 8)
    data['author', 'paper'].edge_index = get_random_edge_index(10, 8, 10)
    data['author', 'paper'].edge_attr = torch.randn(10, 8)

    return data


@pytest.fixture()
def hetero_model():
    return HeteroSAGE
