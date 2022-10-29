import pytest
import torch

from torch_geometric.data import Data
from torch_geometric.explain.explanations import Explanation


def create_random_explanation(data, node_features_mask=True, node_mask=True,
                              edge_mask=True, edge_features_mask=True):
    node_mask = torch.rand(data.x.shape[0]) if node_mask else None
    edge_mask = torch.rand(data.edge_index.shape[1]) if edge_mask else None
    node_features_mask = torch.rand(
        data.x.shape) if node_features_mask else None
    edge_features_mask = torch.rand(
        data.edge_attr.shape) if edge_features_mask else None

    # Create explanation
    return Explanation(x=data.x, edge_index=data.edge_index,
                       node_features_mask=node_features_mask,
                       node_mask=node_mask, edge_mask=edge_mask,
                       edge_features_mask=edge_features_mask)


@pytest.fixture
def data():
    return Data(x=torch.randn(10, 5), edge_index=torch.randint(0, 10, (2, 20)),
                edge_attr=torch.randn(20, 3), target=torch.randn(10, 1))


@pytest.mark.parametrize("node_mask", [True, False])
@pytest.mark.parametrize("edge_mask", [True, False])
@pytest.mark.parametrize("node_features_mask", [True, False])
@pytest.mark.parametrize("edge_features_mask", [True, False])
def test_available_explanations(
    data,
    node_mask,
    edge_mask,
    node_features_mask,
    edge_features_mask,
):
    theoretically_there = []
    if node_mask:
        theoretically_there.append("node_mask")
    if edge_mask:
        theoretically_there.append("edge_mask")
    if node_features_mask:
        theoretically_there.append("node_features_mask")
    if edge_features_mask:
        theoretically_there.append("edge_features_mask")

    explanation = create_random_explanation(data, node_features_mask,
                                            node_mask, edge_mask,
                                            edge_features_mask)

    assert set(explanation.available_explanations) == set(theoretically_there)
