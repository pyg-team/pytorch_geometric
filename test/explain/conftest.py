from typing import Callable, Optional

import pytest
import torch

from torch_geometric.data import Data, HeteroData
from torch_geometric.explain import Explanation
from torch_geometric.explain.config import MaskType
from torch_geometric.nn import SAGEConv, to_hetero


def get_edge_index(num_src_nodes, num_dst_nodes, num_edges):
    row = torch.randint(num_src_nodes, (num_edges, ), dtype=torch.long)
    col = torch.randint(num_dst_nodes, (num_edges, ), dtype=torch.long)
    return torch.stack([row, col], dim=0)


@pytest.fixture()
def data():
    return Data(
        x=torch.randn(4, 3),
        edge_index=get_edge_index(4, 4, num_edges=6),
        edge_attr=torch.randn(6, 3),
    )


@pytest.fixture()
def hetero_data():
    data = HeteroData()
    data['paper'].x = torch.randn(8, 16)
    data['author'].x = torch.randn(10, 8)

    data['paper', 'paper'].edge_index = get_edge_index(8, 8, num_edges=10)
    data['paper', 'paper'].edge_attr = torch.randn(10, 16)
    data['paper', 'author'].edge_index = get_edge_index(8, 10, num_edges=10)
    data['paper', 'author'].edge_attr = torch.randn(10, 8)
    data['author', 'paper'].edge_index = get_edge_index(10, 8, num_edges=10)
    data['author', 'paper'].edge_attr = torch.randn(10, 8)

    return data


@pytest.fixture()
def hetero_model():
    return HeteroSAGE


class GraphSAGE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), 32)
        self.conv2 = SAGEConv((-1, -1), 32)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


class HeteroSAGE(torch.nn.Module):
    def __init__(self, metadata):
        super().__init__()
        self.graph_sage = to_hetero(GraphSAGE(), metadata, debug=False)
        self.lin = torch.nn.Linear(32, 1)

    def forward(self, x_dict, edge_index_dict,
                additonal_arg=None) -> torch.Tensor:
        return self.lin(self.graph_sage(x_dict, edge_index_dict)['paper'])


@pytest.fixture()
def check_explanation() -> Callable:
    return _check_explanation


def _check_explanation(
    explanation: Explanation,
    node_mask_type: Optional[MaskType],
    edge_mask_type: Optional[MaskType],
):
    if node_mask_type == MaskType.attributes:
        assert explanation.node_mask.size() == explanation.x.size()
        assert explanation.node_mask.min() >= 0
        assert explanation.node_mask.max() <= 1
    elif node_mask_type == MaskType.object:
        assert explanation.node_mask.size() == (explanation.num_nodes, 1)
        assert explanation.node_mask.min() >= 0
        assert explanation.node_mask.max() <= 1
    elif node_mask_type == MaskType.common_attributes:
        assert explanation.node_mask.size() == (1, explanation.num_features)
        assert explanation.node_mask.min() >= 0
        assert explanation.node_mask.max() <= 1
    elif node_mask_type is None:
        assert 'node_mask' not in explanation

    if edge_mask_type == MaskType.object:
        assert explanation.edge_mask.size() == (explanation.num_edges, )
        assert explanation.edge_mask.min() >= 0
        assert explanation.edge_mask.max() <= 1
    elif edge_mask_type is None:
        assert 'edge_mask' not in explanation
