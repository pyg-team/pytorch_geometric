import pytest
import torch
import torch.nn.functional as F
from torch.nn import Linear

from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import GNNExplainer
from torch_geometric.explain.config import (
    MaskType,
    ModelReturnType,
    ModelTaskLevel,
)
from torch_geometric.nn import GATConv, GCNConv, global_add_pool
from torch_geometric.testing import is_full_test


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, 7)

    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x.log_softmax(dim=1)


class GAT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GATConv(3, 16, heads=2, concat=True)
        self.conv2 = GATConv(2 * 16, 7, heads=2, concat=False)

    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x.log_softmax(dim=1)


class GNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, 16)
        self.lin = Linear(16, 7)

    def forward(self, x, edge_index, edge_attr, batch):
        # edge_attr is for checking if explain can
        # pass arguments in the right oder
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_add_pool(x, batch)
        x = self.lin(x)
        return x.log_softmax(dim=1)


return_types = [ModelReturnType.log_probs, ModelReturnType.raw]
node_mask_types = [MaskType.attributes]
edge_mask_types = [MaskType.object, None]


@pytest.mark.parametrize('edge_mask_type', edge_mask_types)
@pytest.mark.parametrize('return_type', return_types)
@pytest.mark.parametrize('node_mask_type', node_mask_types)
@pytest.mark.parametrize('model', [GCN(), GAT()])
def test_gnn_explainer_explain_node(model, return_type, node_mask_type,
                                    edge_mask_type):
    explainer = GNNExplainer()

    x = torch.randn(8, 3)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
                               [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6]])
    out = model(x, edge_index)
    if return_type == ModelReturnType.log_probs:
        out = out.log().argmax(dim=-1)

    explanation = explainer(x=x, edge_index=edge_index, model=model,
                            target=out, target_index=0,
                            task_level=ModelTaskLevel.node,
                            return_type=return_type,
                            node_mask_type=node_mask_type,
                            edge_mask_type=edge_mask_type, index=2)

    if node_mask_type == MaskType.attributes:
        assert explanation.node_features_mask.shape == x.shape
        assert explanation.node_features_mask.min() >= 0
        assert explanation.node_features_mask.max() <= 1
    if edge_mask_type == MaskType.object:
        assert explanation.edge_mask.shape == (edge_index.shape[1], )
        assert explanation.edge_mask.min() >= 0
        assert explanation.edge_mask.max() <= 1

    if is_full_test():
        jit = torch.jit.export(explainer)

        explanation = jit(x=x, edge_index=edge_index, model=model, target=out,
                          target_index=2, task_level=ModelTaskLevel.node,
                          return_type=return_type,
                          node_mask_type=node_mask_type,
                          edge_mask_type=edge_mask_type)

        if node_mask_type == MaskType.attributes:
            assert explanation.node_features_mask.shape == x.shape
            assert explanation.node_features_mask.min() >= 0
            assert explanation.node_features_mask.max() <= 1
            assert explanation.edge_mask.shape == (edge_index.shape[1], )
            assert explanation.edge_mask.min() >= 0
            assert explanation.edge_mask.max() <= 1


@pytest.mark.parametrize('edge_mask_type', edge_mask_types)
@pytest.mark.parametrize('return_type', return_types)
@pytest.mark.parametrize('node_mask_type', node_mask_types)
@pytest.mark.parametrize('model', [GNN()])
def test_gnn_explainer_explain_graph(
    model,
    return_type,
    node_mask_type,
    edge_mask_type,
):
    explainer = GNNExplainer()

    x = torch.randn(8, 3)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 4, 5, 5, 6, 6, 7],
                               [1, 0, 2, 1, 3, 2, 5, 4, 6, 5, 7, 6]])
    edge_attr = torch.randn(edge_index.shape[1], 2)
    y = model(x, edge_index, edge_attr, batch=None)
    if return_type == ModelReturnType.log_probs:
        y = y.log().argmax(dim=-1)

    explanation = explainer(x=x, edge_index=edge_index, model=model, target=y,
                            target_index=0, task_level=ModelTaskLevel.graph,
                            return_type=return_type,
                            node_mask_type=node_mask_type,
                            edge_mask_type=edge_mask_type, edge_attr=edge_attr)

    if node_mask_type == MaskType.attributes:
        assert explanation.node_features_mask.shape == x.shape
        assert explanation.node_features_mask.min() >= 0
        assert explanation.node_features_mask.max() <= 1
    if edge_mask_type == MaskType.object:
        assert explanation.edge_mask.shape == (edge_index.shape[1], )
        assert explanation.edge_mask.min() >= 0
        assert explanation.edge_mask.max() <= 1

    if is_full_test():
        jit = torch.jit.export(explainer)

        explanation = jit(x=x, edge_index=edge_index, model=model, target=y,
                          target_index=0, task_level=ModelTaskLevel.graph,
                          return_type=return_type,
                          node_mask_type=node_mask_type,
                          edge_mask_type=edge_mask_type)

        if node_mask_type == MaskType.attributes:
            assert explanation.node_features_mask.shape == x.shape
            assert explanation.node_features_mask.min() >= 0
            assert explanation.node_features_mask.max() <= 1
        if edge_mask_type == MaskType.object:
            assert explanation.edge_mask.shape == (edge_index.shape[1], )
            assert explanation.edge_mask.min() >= 0
            assert explanation.edge_mask.max() <= 1


@pytest.mark.parametrize('return_type', return_types)
@pytest.mark.parametrize('model', [GNN()])
@pytest.mark.parametrize("explanation_type", ["model", "phenomenon"])
@pytest.mark.parametrize("task_level",
                         [ModelTaskLevel.node, ModelTaskLevel.graph])
@pytest.mark.parametrize("edge_mask_type", [None, MaskType.object])
def test_gnn_exlainer_with_meta_explainer(return_type, model, explanation_type,
                                          task_level, edge_mask_type):
    algo = GNNExplainer()
    explainer = Explainer(explanation_algorithm=algo, model=model,
                          model_return_type=return_type,
                          explanation_type=explanation_type,
                          node_mask_type="attributes",
                          edge_mask_type=edge_mask_type, task_level=task_level)
    x = torch.randn(8, 3)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
                               [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6]])
    index = 2 if task_level == ModelTaskLevel.node else None
    out = explainer.get_prediction(x=x, edge_index=edge_index, batch=None,
                                   edge_attr=None)
    explanation = explainer(x=x, edge_index=edge_index, edge_attr=out,
                            target=out, batch=None, index=index)

    assert explanation.node_features_mask.shape == x.shape
    assert explanation.node_features_mask.min() >= 0
    assert explanation.node_features_mask.max() <= 1
    if edge_mask_type == MaskType.object:
        assert explanation.edge_mask.shape == (edge_index.shape[1], )
        assert explanation.edge_mask.min() >= 0
        assert explanation.edge_mask.max() <= 1
