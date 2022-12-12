import pytest
import torch

from torch_geometric.data.hetero_data import HeteroData
from torch_geometric.explain import Explainer, ExplainerAlgorithm, ModelConfig
from torch_geometric.nn import SAGEConv, to_hetero


class HomoExplainerAlgorithm(ExplainerAlgorithm):
    def supports(self) -> bool:
        return not (self._is_hetero is True)


class HeteroExplainerAlgorithm(ExplainerAlgorithm):
    def supports(self) -> bool:
        # Must be careful to allow None, which is the default value
        return not (self._is_hetero is False)


def get_edge_index(num_src_nodes, num_dst_nodes, num_edges):
    row = torch.randint(num_src_nodes, (num_edges, ), dtype=torch.long)
    col = torch.randint(num_dst_nodes, (num_edges, ), dtype=torch.long)
    return torch.stack([row, col], dim=0)


def hetero_data():
    data = HeteroData()
    data['paper'].x = torch.randn(8, 16)
    data['author'].x = torch.randn(10, 8)
    data['paper', 'paper'].edge_index = get_edge_index(8, 8, 10)
    data['author', 'paper'].edge_index = get_edge_index(10, 8, 10)
    data['paper', 'author'].edge_index = get_edge_index(8, 10, 10)
    return data


class GraphSAGE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), 32)
        self.conv2 = SAGEConv((-1, -1), 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


class HeteroSAGE(torch.nn.Module):
    def __init__(self, metadata):
        super().__init__()
        self.graph_sage = to_hetero(GraphSAGE(), metadata, debug=False)

    def forward(self, x_dict, edge_index_dict, *args) -> torch.Tensor:
        return self.graph_sage(x_dict, edge_index_dict)['paper']


def test_supports_hetero():
    data = hetero_data()
    model = HeteroSAGE(data.metadata())

    explainer = Explainer(
        model=model, algorithm=HomoExplainerAlgorithm(),
        explanation_type="model", node_mask_type="object", edge_mask_type=None,
        model_config=ModelConfig(
            mode="regression",
            task_level="node",
        ))

    with pytest.raises(ValueError):
        explainer(data.x_dict, data.edge_index_dict)

    explainer.algorithm = HeteroExplainerAlgorithm()

    explainer(data.x_dict, data.edge_index_dict)
