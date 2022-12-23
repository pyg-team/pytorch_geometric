import pytest
import torch

from torch_geometric.explain import Explainer  # , Explanation
from torch_geometric.explain.algorithm import PGMExplainer
from torch_geometric.explain.config import ModelConfig
from torch_geometric.nn import GCNConv, global_add_pool


class GCN(torch.nn.Module):
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.model_config = model_config

        if model_config.mode.value == 'multiclass_classification':
            out_channels = 7
        else:
            out_channels = 1

        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, out_channels)

    def forward(self, x, edge_index, edge_weight):

        return x.log_softmax(dim=1)

    def forward(self, x, edge_index, edge_weight, batch=None):
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = self.conv2(x, edge_index, edge_weight).relu()

        if self.model_config.task_level.value == 'graph':
            x = global_add_pool(x, batch)

        if self.model_config.mode.value == 'binary_classification':
            if self.model_config.return_type.value == 'probs':
                x = x.sigmoid()
        elif self.model_config.mode.value == 'multiclass_classification':
            if self.model_config.return_type.value == 'probs':
                x = x.softmax(dim=-1)
            elif self.model_config.return_type.value == 'log_probs':
                x = x.log_softmax(dim=-1)

        return x


x = torch.randn(8, 3)
edge_index = torch.tensor([
    [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
    [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6],
])
batch = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2])
edge_label_index = torch.tensor([[0, 1, 2], [3, 4, 5]])


@pytest.mark.parametrize('index', [None, 2, torch.arange(3)])
def test_pgm_explainer_node_binary_classification(
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

    model = GCN(model_config)

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
        algorithm=PGMExplainer(),
        explanation_type=explanation_type,
        node_mask_type=node_mask_type,
        edge_mask_type=edge_mask_type,
        model_config=model_config,
    )

    explanation = explainer(
        x,
        edge_index,
        target=target,
        index=index,
        target_index=0 if multi_output else None,
        batch=batch,
        edge_label_index=edge_label_index,
    )

    assert explanation.node_mask is not None


def test_pgm_explainer_graph_classification(
    edge_mask_type,
    node_mask_type,
    explanation_type,
    task_level,
    return_type,
    index,
    multi_output,
):
    pass
