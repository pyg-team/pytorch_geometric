import pytest
import torch

from torch_geometric.explain import AttentionExplainer, Explainer
from torch_geometric.explain.config import ExplanationType, MaskType
from torch_geometric.nn import GATConv, GATv2Conv, TransformerConv


class AttentionGNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GATConv(3, 16, heads=4)
        self.conv2 = GATv2Conv(4 * 16, 16, heads=2)
        self.conv3 = TransformerConv(2 * 16, 7, heads=1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        return x


x = torch.randn(8, 3)
edge_index = torch.tensor([
    [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
    [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6],
])


@pytest.mark.parametrize('index', [None, 2, torch.arange(3)])
def test_attention_explainer(index, check_explanation):
    explainer = Explainer(
        model=AttentionGNN(),
        algorithm=AttentionExplainer(),
        explanation_type='model',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='raw',
        ),
    )

    explanation = explainer(x, edge_index, index=index)
    check_explanation(explanation, None, explainer.edge_mask_type)


@pytest.mark.parametrize('explanation_type', [e for e in ExplanationType])
@pytest.mark.parametrize('node_mask_type', [m for m in MaskType])
def test_attention_explainer_supports(explanation_type, node_mask_type):
    with pytest.raises(ValueError, match="not support the given explanation"):
        Explainer(
            model=AttentionGNN(),
            algorithm=AttentionExplainer(),
            explanation_type=explanation_type,
            node_mask_type=node_mask_type,
            edge_mask_type='object',
            model_config=dict(
                mode='multiclass_classification',
                task_level='node',
                return_type='raw',
            ),
        )
