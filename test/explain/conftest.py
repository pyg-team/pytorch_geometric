from typing import Optional

import pytest
import torch

from torch_geometric.data import Data, HeteroData
from torch_geometric.explain import Explanation, HeteroExplanation
from torch_geometric.explain.config import (
    MaskType,
    ModelConfig,
    ModelMode,
    ModelReturnType,
)
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.testing import get_random_edge_index


@pytest.fixture()
def data():
    return Data(
        x=torch.randn(4, 3),
        edge_index=get_random_edge_index(4, 4, num_edges=6),
        edge_attr=torch.randn(6, 3),
    )


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


class GraphSAGE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), 32)
        self.conv2 = SAGEConv((-1, -1), 32)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


class HeteroSAGE(torch.nn.Module):
    def __init__(self, metadata, model_config: Optional[ModelConfig] = None):
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
        if self.model_config:
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
def check_explanation():
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
            assert explanation.node_mask.size() == (1, explanation.x.size(-1))
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

    return _check_explanation


@pytest.fixture()
def check_explanation_hetero():
    def _check_explanation_hetero(
        explanation: HeteroExplanation,
        node_mask_type: Optional[MaskType],
        edge_mask_type: Optional[MaskType],
        hetero_data: HeteroData,
    ):
        # Validate the explanation
        explanation.validate(raise_on_error=True)

        # Check node masks for different node types
        if node_mask_type is not None:
            for node_type in hetero_data.node_types:
                assert explanation[node_type].get('node_mask') is not None
                assert explanation[node_type].get('node_mask').min() >= 0
                assert explanation[node_type].get('node_mask').max() <= 1

                # Check dimensions based on mask type
                if node_mask_type == MaskType.attributes:
                    mask = explanation[node_type].get('node_mask')
                    assert mask.size() == hetero_data.x_dict[node_type].size()
                elif node_mask_type == MaskType.object:
                    mask = explanation[node_type].get('node_mask')
                    assert mask.size() == (
                        hetero_data.x_dict[node_type].size(0), 1)
                elif node_mask_type == MaskType.common_attributes:
                    mask = explanation[node_type].get('node_mask')
                    assert mask.size() == (
                        1, hetero_data.x_dict[node_type].size(1))

        # Check edge masks for different edge types
        if edge_mask_type is not None:
            for edge_type in hetero_data.edge_types:
                assert explanation[edge_type].get('edge_mask') is not None
                assert explanation[edge_type].get('edge_mask').min() >= 0
                assert explanation[edge_type].get('edge_mask').max() <= 1

    return _check_explanation_hetero
