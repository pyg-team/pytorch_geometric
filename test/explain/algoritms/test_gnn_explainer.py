import pytest
import torch
import torch.nn.functional as F
from torch.nn import Linear

from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import GNNExplainer
from torch_geometric.explain.config import (
    ExplainerConfig,
    MaskType,
    ModelConfig,
    ModelReturnType,
    ModelTaskLevel,
)
from torch_geometric.nn import GATConv, GCNConv, global_add_pool
from torch_geometric.testing import is_full_test


#  TODO create a model for each combination of
#      regression/ classification
#      graph/ node/ (edge)
#         Single target/ multi target
#      return type: raw/ probs/ log_probs
class GCN_multioutput(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, 7)

    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        return self.conv2(x, edge_index)


class GCN_single_output(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, 1)

    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        return self.conv2(x, edge_index)


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


class GCN_multioutput_classification(torch.nn.Module):
    def __init__(self):
        raise NotImplementedError


class GAT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GATConv(3, 16, heads=2, concat=True)
        self.conv2 = GATConv(2 * 16, 7, heads=2, concat=False)

    def forward(self, x, edge_index, batch=None, **kwargs):
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


# type of masks allowed for GNNExplainer
node_mask_types = ["attributes"]
edge_mask_types = ["object", None]
return_types_classification = ["log_probs", "raw", "probs"]
return_types_regression = ["raw"]


@pytest.mark.parametrize('edge_mask_type', edge_mask_types)
@pytest.mark.parametrize('node_mask_type', node_mask_types)
@pytest.mark.parametrize('model', [GCN_single_output])
@pytest.mark.parametrize('return_type', return_types_regression)
def test_gnn_explainer_node_regression_single_output(edge_mask_type,
                                                     node_mask_type, model,
                                                     return_type):
    explainer_config = ExplainerConfig(
        explanation_type="model",
        node_mask_type=node_mask_type,
        edge_mask_type=edge_mask_type,
    )
    model_config = ModelConfig(
        task_level='node',
        return_type=return_type,
        mode="regression",
    )

    explainer = GNNExplainer()
    model = model()

    x = torch.randn(8, 3)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
                               [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6]])
    out = model(x, edge_index)

    # try to explain prediction for node 0
    explanation = explainer(x=x, edge_index=edge_index, model=model,
                            target=out, target_index=0,
                            explainer_config=explainer_config,
                            model_config=model_config, index=0)

    assert explanation.node_features_mask.shape == x.shape
    assert explanation.node_features_mask.min() >= 0
    assert explanation.node_features_mask.max() <= 1

    if edge_mask_type == MaskType.object:
        assert explanation.edge_mask.shape == (edge_index.shape[1], )
        assert explanation.edge_mask.min() >= 0
        assert explanation.edge_mask.max() <= 1


@pytest.mark.parametrize('edge_mask_type', edge_mask_types)
@pytest.mark.parametrize('node_mask_type', node_mask_types)
@pytest.mark.parametrize('model', [GCN_multioutput, GAT])
@pytest.mark.parametrize('return_type', return_types_regression)
def test_gnn_explainer_node_regression_multioutput(edge_mask_type,
                                                   node_mask_type, model,
                                                   return_type):
    explainer_config = ExplainerConfig(
        explanation_type="model",
        node_mask_type=node_mask_type,
        edge_mask_type=edge_mask_type,
    )
    model_config = ModelConfig(
        task_level='node',
        return_type=return_type,
        mode="regression",
    )

    explainer = GNNExplainer()
    model = model()

    x = torch.randn(8, 3)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
                               [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6]])
    out = model(x, edge_index)

    # try to explain prediction for node 0
    explanation = explainer(x=x, edge_index=edge_index, model=model,
                            target=out, target_index=2,
                            explainer_config=explainer_config,
                            model_config=model_config, index=0)

    assert explanation.node_features_mask.shape == x.shape
    assert explanation.node_features_mask.min() >= 0
    assert explanation.node_features_mask.max() <= 1

    if edge_mask_type == MaskType.object:
        assert explanation.edge_mask.shape == (edge_index.shape[1], )
        assert explanation.edge_mask.min() >= 0
        assert explanation.edge_mask.max() <= 1


@pytest.mark.parametrize('edge_mask_type', edge_mask_types)
@pytest.mark.parametrize('node_mask_type', node_mask_types)
@pytest.mark.parametrize('model', [GCN_single_output])
@pytest.mark.parametrize('return_type', return_types_classification)
def test_gnn_explainer_node_classification_single_output(
        edge_mask_type, node_mask_type, model, return_type):
    explainer_config = ExplainerConfig(
        explanation_type="model",
        node_mask_type=node_mask_type,
        edge_mask_type=edge_mask_type,
    )
    model_config = ModelConfig(
        task_level='node',
        return_type=return_type,
        mode="classification",
    )

    explainer = GNNExplainer()
    model = model()

    x = torch.randn(8, 3)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
                               [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6]])
    out = model(x, edge_index).argmax(dim=1)

    # try to explain prediction for node 0
    explanation = explainer(x=x, edge_index=edge_index, model=model,
                            target=out, target_index=0,
                            explainer_config=explainer_config,
                            model_config=model_config, index=0)

    assert explanation.node_features_mask.shape == x.shape
    assert explanation.node_features_mask.min() >= 0
    assert explanation.node_features_mask.max() <= 1

    if edge_mask_type == MaskType.object:
        assert explanation.edge_mask.shape == (edge_index.shape[1], )
        assert explanation.edge_mask.min() >= 0
        assert explanation.edge_mask.max() <= 1


@pytest.mark.parametrize('edge_mask_type', edge_mask_types)
@pytest.mark.parametrize('node_mask_type', node_mask_types)
@pytest.mark.parametrize('model', [GCN_multioutput_classification])
@pytest.mark.parametrize('return_type', return_types_classification)
def test_gnn_explainer_node_multioutput_classification(edge_mask_type,
                                                       node_mask_type, model,
                                                       return_type):
    explainer_config = ExplainerConfig(
        explanation_type="model",
        node_mask_type=node_mask_type,
        edge_mask_type=edge_mask_type,
    )
    model_config = ModelConfig(
        task_level='node',
        return_type=return_type,
        mode="classification",
    )

    explainer = GNNExplainer()
    model = model()

    x = torch.randn(8, 3)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
                               [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6]])
    out = model(x, edge_index)

    # try to explain prediction for output 2  for node 0
    explanation = explainer(x=x, edge_index=edge_index, model=model,
                            target=out, target_index=2,
                            explainer_config=explainer_config,
                            model_config=model_config, index=0)

    assert explanation.node_features_mask.shape == x.shape
    assert explanation.node_features_mask.min() >= 0
    assert explanation.node_features_mask.max() <= 1

    if edge_mask_type == MaskType.object:
        assert explanation.edge_mask.shape == (edge_index.shape[1], )
        assert explanation.edge_mask.min() >= 0
        assert explanation.edge_mask.max() <= 1


def nope_test_gnn_explainer_graph_classification():
    raise NotImplementedError


def nope_test_gnn_explainer_graph_regression():
    raise NotImplementedError


def nope_test_gnn_explainer_with_meta_explainer():
    raise NotImplementedError


return_types = return_types_classification


@pytest.mark.parametrize('edge_mask_type', edge_mask_types)
@pytest.mark.parametrize('return_type', return_types)
@pytest.mark.parametrize('node_mask_type', node_mask_types)
@pytest.mark.parametrize('model', [GCN, GAT])
def nope_test_gnn_explainer_explain_node(model, return_type, node_mask_type,
                                         edge_mask_type):
    explainer = GNNExplainer()
    model = model()

    x = torch.randn(8, 3)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
                               [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6]])

    out = model(x, edge_index)
    if return_type == ModelReturnType.log_probs:
        out = out.log().argmax(dim=-1)

    explainer_config = ExplainerConfig(
        explanation_type="model",
        node_mask_type=node_mask_type,
        edge_mask_type=edge_mask_type,
    )
    model_config = ModelConfig(
        task_level='node',
        return_type=return_type,
        mode='classification'
        if return_type == ModelReturnType.log_probs else 'regression',
    )

    explanation = explainer(x=x, edge_index=edge_index, model=model,
                            target=out, target_index=2,
                            explainer_config=explainer_config,
                            model_config=model_config, index=0)

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
@pytest.mark.parametrize('model', [GNN])
def nope_test_gnn_explainer_explain_graph(
    model,
    return_type,
    node_mask_type,
    edge_mask_type,
):
    explainer = GNNExplainer()
    model = model()

    x = torch.randn(8, 3)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 4, 5, 5, 6, 6, 7],
                               [1, 0, 2, 1, 3, 2, 5, 4, 6, 5, 7, 6]])
    edge_attr = torch.randn(edge_index.shape[1], 2)
    y = model(x, edge_index, edge_attr, batch=None)
    if return_type == ModelReturnType.log_probs:
        y = y.log().argmax(dim=-1)

    explainer_config = ExplainerConfig(
        explanation_type="model",
        node_mask_type=node_mask_type,
        edge_mask_type=edge_mask_type,
    )
    model_config = ModelConfig(
        task_level='graph',
        return_type=return_type,
        mode='classification'
        if return_type == ModelReturnType.log_probs else 'regression',
    )

    explanation = explainer(x=x, edge_index=edge_index, model=model, target=y,
                            target_index=0, model_config=model_config,
                            explainer_config=explainer_config,
                            edge_attr=edge_attr)

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
@pytest.mark.parametrize('model', [GNN, GAT])
@pytest.mark.parametrize("explanation_type", ["model", "phenomenon"])
@pytest.mark.parametrize("edge_mask_type", [None, MaskType.object])
def nope_test_gnn_exlainer_with_meta_explainer(return_type, model,
                                               explanation_type,
                                               edge_mask_type):
    model = model()
    algo = GNNExplainer()
    task_level = ModelTaskLevel.node if isinstance(
        model, GAT) else ModelTaskLevel.graph
    explainer_config = ExplainerConfig(
        explanation_type=explanation_type,
        node_mask_type="attributes",
        edge_mask_type=edge_mask_type,
    )
    model_config = ModelConfig(
        task_level=task_level,
        return_type=return_type,
        mode='classification',
    )

    explainer = Explainer(algorithm=algo, model=model,
                          explainer_config=explainer_config,
                          model_config=model_config)

    x = torch.randn(8, 3)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
                               [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6]])
    index = 2 if task_level == ModelTaskLevel.node else None
    out = explainer.get_prediction(x=x, edge_index=edge_index, batch=None,
                                   edge_attr=None)
    explanation = explainer(x=x, edge_index=edge_index, edge_attr=out,
                            target=out, batch=None, target_index=index,
                            index=index)

    assert explanation.node_features_mask.shape == x.shape
    assert explanation.node_features_mask.min() >= 0
    assert explanation.node_features_mask.max() <= 1
    if edge_mask_type == MaskType.object:
        assert explanation.edge_mask.shape == (edge_index.shape[1], )
        assert explanation.edge_mask.min() >= 0
        assert explanation.edge_mask.max() <= 1
