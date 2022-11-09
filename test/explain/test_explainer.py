import pytest
import torch

from torch_geometric.data import Data
from torch_geometric.explain import (
    DummyExplainer,
    Explainer,
    ExplainerAlgorithm,
    Explanation,
)
from torch_geometric.explain.config import (
    ExplanationType,
    ModelReturnType,
    ModelTaskLevel,
    ThresholdType,
)

g = Data(
    x=torch.randn(8, 3, requires_grad=True),
    edge_index=torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
                             [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6]]),
    edge_attr=torch.randn(14, 3), target=torch.randn(14, 1))


def dummyExplanation(g) -> Explanation:
    n_nodes = g.x.shape[0]
    n_edges = g.edge_index.shape[1]
    f_nodes = g.x.shape[1]
    f_edges = g.edge_attr.shape[1]
    big_max = torch.tensor([n_nodes * f_nodes, n_edges * f_edges]).max() + 1
    masks = {
        "node_mask":
        torch.arange(1., 1. + n_nodes, dtype=torch.float) / big_max,
        "edge_mask":
        torch.arange(1., 1. + n_edges, dtype=torch.float) / big_max,
        "node_features_mask":
        torch.arange(1., 1. + f_nodes * n_nodes, dtype=torch.float).reshape(
            n_nodes, f_nodes) / big_max,
        "edge_features_mask":
        torch.arange(1., 1. + f_edges * n_edges, dtype=torch.float).reshape(
            n_edges, f_edges) / big_max
    }
    return Explanation(edge_index=g.edge_index, x=g.x, edge_attr=g.edge_attr,
                       **masks)


def constantExplanation(g) -> Explanation:
    n_nodes = g.x.shape[0]
    n_edges = g.edge_index.shape[1]
    f_nodes = g.x.shape[1]
    f_edges = g.edge_attr.shape[1]
    masks = {
        "node_mask":
        torch.ones(n_nodes),
        "edge_mask":
        torch.ones(n_edges),
        "node_features_mask":
        torch.ones(f_nodes * n_nodes).reshape(n_nodes, f_nodes),
        "edge_features_mask":
        torch.ones(f_edges * n_edges).reshape(n_edges, f_edges)
    }
    return Explanation(edge_index=g.edge_index, x=g.x, edge_attr=g.edge_attr,
                       **masks)


class DummyModel(torch.nn.Module):
    def __init__(self, out_dim: int = 1) -> None:
        super().__init__()
        self.out_dim = out_dim

    def forward(self, x, **kwargs):
        return x.mean().repeat(self.out_dim)


class DummyExplainer(ExplainerAlgorithm):
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        **kwargs,
    ) -> Data:
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def supports(
        self,
        *args,
    ):
        return True, None

    def loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.mse_loss(y_hat, y)


@pytest.fixture(scope="module")
def dummyexplainer():
    return DummyExplainer()


@pytest.fixture(scope="module")
def dummyInput() -> Data:
    return g


@pytest.fixture(scope="module")
def explanation_algorithm() -> ExplainerAlgorithm:
    return DummyExplainer()


models = [DummyModel(), DummyModel(2)]
thresholds = [x.name for x in list(ThresholdType)]
model_return_types = [x.name for x in list(ModelReturnType)]
explanation_types = [x.name for x in list(ExplanationType)]
task_levels = [x.name for x in list(ModelTaskLevel)]


@pytest.mark.parametrize("model", models)
def test_get_prediction(dummyexplainer, dummyInput, model):
    explainer = Explainer(explanation_algorithm=dummyexplainer, model=model,
                          model_return_type="regression", task_level="node",
                          explanation_type="phenomenon",
                          node_mask_type="object", threshold="none")
    assert explainer.get_prediction(
        x=dummyInput.x, edge_index=dummyInput.edge_index,
        edge_attr=dummyInput.edge_attr).shape == (model.out_dim, )


@pytest.mark.parametrize("target", [None, torch.randn(2)])
@pytest.mark.parametrize("explanation_type", explanation_types)
def test_forward_target_switch(dummyInput, target, explanation_type):
    explainer = Explainer(explanation_algorithm=DummyExplainer(),
                          model=DummyModel(), model_return_type="regression",
                          task_level="graph",
                          explanation_type=explanation_type,
                          node_mask_type="attributes", threshold="none")
    if target is None and explanation_type == "phenomenon":
        with pytest.raises(ValueError):
            explainer(x=dummyInput.x, edge_index=dummyInput.edge_index,
                      edge_attr=dummyInput.edge_attr, target=target)
    else:
        assert explainer(
            x=dummyInput.x, edge_index=dummyInput.edge_index,
            edge_attr=dummyInput.edge_attr,
            target=target).node_features_mask.shape == dummyInput.x.shape


@pytest.mark.parametrize("threshold_value", [0.2, 0.5, 0.8])
def test_hard_threshold(dummyInput, threshold_value):
    explainer = Explainer(explanation_algorithm=DummyExplainer(),
                          model=DummyModel(), model_return_type="regression",
                          task_level="graph", explanation_type="phenomenon",
                          node_mask_type="both", threshold="hard",
                          edge_mask_type="both",
                          threshold_value=threshold_value)
    explanations_raw = explainer.explanation_algorithm(dummyInput.x,
                                                       dummyInput.edge_index,
                                                       dummyInput.edge_attr,
                                                       target=None,
                                                       target_index=0)
    explanation_processed = explainer._threshold(explanations_raw)
    assert explanation_processed.node_features_mask.shape == \
        dummyInput.x.shape
    for exp in explanations_raw.available_explanations:
        assert torch.all(explanations_raw[exp] >= 0)
        assert torch.all(explanations_raw[exp] <= 1)
        assert torch.any((explanation_processed[exp] == 0)
                         | (explanation_processed[exp] == 1))
        assert torch.all(
            torch.masked_select(explanation_processed[exp],
                                explanation_processed[exp].bool()) >
            threshold_value)
        assert torch.all(
            torch.masked_select(explanation_processed[exp], (
                explanation_processed[exp] == 0)) <= threshold_value)


@pytest.mark.parametrize("threshold_value", list(range(1, 10)))
@pytest.mark.parametrize(
    "explanation",
    [constantExplanation(g), dummyExplanation(g)])
def test_topk_threshold(explanation, threshold_value):
    explainer = Explainer(explanation_algorithm=DummyExplainer(),
                          model=DummyModel(), model_return_type="regression",
                          task_level="graph", explanation_type="phenomenon",
                          threshold="topk", node_mask_type="both",
                          edge_mask_type="both",
                          threshold_value=threshold_value)

    exp_processed = explainer._threshold(explanation)
    for exp in explanation.available_explanations:

        if threshold_value < exp_processed[exp].numel():
            assert torch.sum(exp_processed[exp] > 0) == threshold_value
            assert torch.all(
                torch.masked_select(explanation[exp], exp_processed[exp] > 0).
                min() >= torch.masked_select(explanation[exp],
                                             (exp_processed[exp] == 0)).max())
        else:
            assert torch.all(exp_processed[exp] == explanation[exp])


@pytest.mark.parametrize("threshold_value", list(range(1, 10)))
@pytest.mark.parametrize(
    "explanation",
    [constantExplanation(g), dummyExplanation(g)])
def test_topk_hard_threshold(explanation, threshold_value):
    explainer = Explainer(explanation_algorithm=DummyExplainer(),
                          model=DummyModel(), model_return_type="regression",
                          task_level="graph", explanation_type="phenomenon",
                          node_mask_type="attributes", threshold="topk_hard",
                          threshold_value=threshold_value)

    exp_processed = explainer._threshold(explanation)
    for exp in explanation.available_explanations:

        if threshold_value < exp_processed[exp].numel():
            assert torch.sum(exp_processed[exp] > 0) == threshold_value
        else:
            assert torch.sum(
                exp_processed[exp] > 0) == exp_processed[exp].numel()
