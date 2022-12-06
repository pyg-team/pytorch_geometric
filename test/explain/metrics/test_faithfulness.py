import pytest
import torch

from torch_geometric.data import Data
from torch_geometric.explain import Explanation
from torch_geometric.explain.metrics.faithfulness import get_faithfulness
from torch_geometric.nn.models import GCN


@pytest.fixture
def data():
    return Data(
        x=torch.randn(4, 3),
        edge_index=torch.tensor([
            [0, 0, 0, 1, 1, 2],
            [1, 2, 3, 2, 3, 3],
        ]),
        edge_attr=torch.randn(6, 3),
    )


def create_random_explanation(
    data: Data,
    node_mask: bool = True,
    edge_mask: bool = True,
    node_feat_mask: bool = True,
    edge_feat_mask: bool = True,
):
    node_mask = torch.rand(data.x.size(0)) if node_mask else None
    edge_mask = torch.rand(data.edge_index.size(1)) if edge_mask else None
    node_feat_mask = torch.rand_like(data.x) if node_feat_mask else None
    edge_feat_mask = (torch.rand_like(data.edge_attr)
                      if edge_feat_mask else None)

    return Explanation(  # Create explanation.
        node_mask=node_mask,
        edge_mask=edge_mask,
        node_feat_mask=node_feat_mask,
        edge_feat_mask=edge_feat_mask,
        x=data.x,
        edge_index=data.edge_index,
        edge_attr=data.edge_attr,
    )


model = GCN(in_channels=3, hidden_channels=4, num_layers=2, out_channels=2)


@pytest.mark.parametrize('topk', [1, 2])
@pytest.mark.parametrize('node_idx', [0, 1])
@pytest.mark.parametrize('node_mask', [True, False])
@pytest.mark.parametrize('edge_mask', [True, False])
@pytest.mark.parametrize('node_feat_mask', [True, False])
@pytest.mark.parametrize('edge_feat_mask', [True, False])
def test_faithfulness(data, topk, node_idx, node_mask, edge_mask,
                      node_feat_mask, edge_feat_mask):
    explanation = create_random_explanation(
        data,
        node_mask=node_mask,
        edge_mask=edge_mask,
        node_feat_mask=node_feat_mask,
        edge_feat_mask=edge_feat_mask,
    )

    val1, val2 = get_faithfulness(data=data, explanation=explanation,
                                  model=model, topk=topk, node_idx=node_idx)

    assert val1 is None or (val1 >= 0)
    assert val2 is None or (val2 >= 0)
