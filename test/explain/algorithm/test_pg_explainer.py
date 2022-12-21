import pytest
import torch

from torch_geometric.explain import Explainer, Explanation, PGExplainer
from torch_geometric.explain.config import MaskType, ModelConfig
from torch_geometric.nn import GCNConv, global_add_pool


class GCN(torch.nn.Module):
    def __init__(self, model_config: ModelConfig, multi_output: bool = False):
        super().__init__()
        self.model_config = model_config
        self.multi_output = multi_output

        if model_config.mode.value == 'multiclass_classification':
            out_channels = 7
        else:
            out_channels = 1

        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, out_channels)

    def forward(self, x, edge_index, batch=None, edge_label_index=None):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)

        if self.model_config.task_level.value == 'graph':
            x = global_add_pool(x, batch)
        elif self.model_config.task_level.value == 'edge':
            assert edge_label_index is not None
            x = x[edge_label_index[0]] * x[edge_label_index[1]]

        if self.model_config.mode.value == 'binary_classification':
            if self.model_config.return_type.value == 'probs':
                x = x.sigmoid()
        elif self.model_config.mode.value == 'multiclass_classification':
            if self.model_config.return_type.value == 'probs':
                x = x.softmax(dim=-1)
            elif self.model_config.return_type.value == 'log_probs':
                x = x.log_softmax(dim=-1)

        return torch.stack([x, x], dim=0) if self.multi_output else x


def check_explanation(
    edge_mask_type: MaskType,
    node_mask_type: MaskType,
    explanation: Explanation,
):
    if node_mask_type == MaskType.attributes:
        assert explanation.node_feat_mask.size() == explanation.x.size()
        assert explanation.node_feat_mask.min() >= 0
        assert explanation.node_feat_mask.max() <= 1
    elif node_mask_type == MaskType.object:
        assert explanation.node_mask.size() == (explanation.num_nodes, )
        assert explanation.node_mask.min() >= 0
        assert explanation.node_mask.max() <= 1
    elif node_mask_type == MaskType.common_attributes:
        assert explanation.node_feat_mask.size() == explanation.x.size()
        assert explanation.node_feat_mask.min() >= 0
        assert explanation.node_feat_mask.max() <= 1
        assert torch.allclose(explanation.node_feat_mask[0],
                              explanation.node_feat_mask[1])


node_mask_types = [None]
edge_mask_types = [
    MaskType.object,
]

x = torch.randn(8, 3)
edge_index = torch.tensor([
    [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
    [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6],
])
batch = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2])
edge_label_index = torch.tensor([[0, 1, 2], [3, 4, 5]])


@pytest.mark.parametrize('edge_mask_type', edge_mask_types)
@pytest.mark.parametrize('node_mask_type', node_mask_types)
@pytest.mark.parametrize('explanation_type', ['model', 'phenomenon'])
@pytest.mark.parametrize('task_level', ['node', 'graph'])
@pytest.mark.parametrize('return_type', ['probs', 'raw'])
@pytest.mark.parametrize('index', [1, 2])
@pytest.mark.parametrize('multi_output', [False])
def test_pg_explainer_binary_classification(
    edge_mask_type,
    node_mask_type,
    explanation_type,
    task_level,
    return_type,
    index,
    multi_output,
):
    model_config = ModelConfig(
        mode='binary_classification',
        task_level=task_level,
        return_type=return_type,
    )

    model = GCN(model_config, multi_output)

    target = None
    if explanation_type == 'phenomenon':
        with torch.no_grad():
            out = model(x, edge_index, batch, edge_label_index)
            if model_config.return_type.value == 'raw':
                target = (out > 0).long().view(-1)
            if model_config.return_type.value == 'probs':
                target = (out > 0.5).long().view(-1)

    explainer = Explainer(
        model=model,
        algorithm=PGExplainer(epochs=2),
        explanation_type=explanation_type,
        node_mask_type=node_mask_type,
        edge_mask_type=edge_mask_type,
        model_config=model_config,
    )
    explainer.train_explainer_algorithm(x, edge_index, target=target,
                                        batch=batch)

    explanation = explainer(
        x,
        edge_index,
        target=target,
        index=index,
        batch=batch,
        edge_label_index=edge_label_index,
    )

    check_explanation(edge_mask_type, node_mask_type, explanation)


@pytest.mark.parametrize('edge_mask_type', edge_mask_types)
@pytest.mark.parametrize('node_mask_type', node_mask_types)
@pytest.mark.parametrize('explanation_type', ['model', 'phenomenon'])
@pytest.mark.parametrize('task_level', ['node', 'graph'])
@pytest.mark.parametrize('index', [2, 3])
@pytest.mark.parametrize('multi_output', [False])
def test_pg_explainer_regression(
    edge_mask_type,
    node_mask_type,
    explanation_type,
    task_level,
    index,
    multi_output,
):
    model_config = ModelConfig(
        mode='regression',
        task_level=task_level,
    )

    model = GCN(model_config, multi_output)

    target = None
    if explanation_type == 'phenomenon':
        with torch.no_grad():
            target = model(x, edge_index, batch, edge_label_index)

    explainer = Explainer(
        model=model,
        algorithm=PGExplainer(epochs=2),
        explanation_type=explanation_type,
        node_mask_type=node_mask_type,
        edge_mask_type=edge_mask_type,
        model_config=model_config,
    )
    explainer.train_explainer_algorithm(x, edge_index, target=target,
                                        batch=batch)
    explanation = explainer(
        x,
        edge_index,
        target=target,
        index=index,
        batch=batch,
    )

    check_explanation(edge_mask_type, node_mask_type, explanation)
