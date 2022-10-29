import pytest
import torch

from torch_geometric.data import Data
from torch_geometric.explain.algorithm import RandomExplainer
from torch_geometric.explain.base import ExplainerAlgorithm
from torch_geometric.explain.explainer import (
    Explainer,
    ExplainerConfig,
    ModelConfig,
    Threshold,
)
from torch_geometric.explain.utils import Interface
from torch_geometric.nn import GAT, GCN


class DummyModel(torch.nn.Module):
    def __init__(self, out_dim: int = 1) -> None:
        super().__init__()
        self.out_dim = out_dim

    def forward(self, g: Data, **kwargs):
        return g.x.mean().repeat(self.out_dim)


class DummyModelNotCompatible(torch.nn.Module):
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return x


class DummyExplainer(ExplainerAlgorithm):
    def forward(
        self,
        g: Data,
        **kwargs,
    ) -> Data:
        return g

    def explain(
        self,
        g: Data,
        **kwargs,
    ) -> Data:
        return g

    def supports(
        self,
        explanation_type: str,
        mask_type: str,
    ) -> bool:
        return True

    def loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.mse_loss(y_hat, y)


@pytest.fixture(scope="module")
def dummyexplainer():
    return DummyExplainer()


@pytest.fixture(scope="module")
def dummyInput() -> Data:
    return Data(
        x=torch.randn(8, 3, requires_grad=True),
        edge_index=torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
                                 [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6]]),
        edge_attr=torch.randn(14, 3), target=torch.randn(14, 1))


@pytest.fixture(scope="module")
def explanation_algorithm() -> ExplainerAlgorithm:
    return RandomExplainer()


class GCNCcompatible(GCN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.interface = Interface(graph_to_inputs=lambda x: {
            "x": x.x,
            "edge_index": x.edge_index
        })

    def forward(self, g, **kwargs):
        return super().forward(**self.interface.graph_to_inputs(g))


class GATCcompatible(GAT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.interface = Interface(graph_to_inputs=lambda x: {
            "x": x.x,
            "edge_index": x.edge_index
        })

    def forward(self, g, **kwargs):
        return super().forward(**self.interface.graph_to_inputs(g))


models = [DummyModel(), DummyModel(2)]
model_return_types = ModelConfig._valid_model_return_type
explanation_types = ExplainerConfig._valid_explanation_type
mask_types = ExplainerConfig._valid_mask_type
thresholds = Threshold._valid_type
task_levels = ModelConfig._valid_task


@pytest.mark.parametrize("model", models[:1])
@pytest.mark.parametrize("model_return_type", model_return_types[:1])
@pytest.mark.parametrize("task_level", task_levels[:1])
@pytest.mark.parametrize("explanation_type", explanation_types[:1])
@pytest.mark.parametrize("mask_type", mask_types[:1])
@pytest.mark.parametrize("threshold_pairs", [("none", 0, True),
                                             ("hard", 0.5, True),
                                             ("hard", 1.1, False),
                                             ("hard", -1, False),
                                             ("topk", 1, True),
                                             ("topk", 0, False),
                                             ("topk", -1, False),
                                             ("topk", 0.5, False),
                                             ("connected", 1, True),
                                             ("connected", 0, False),
                                             ("connected", -1, False),
                                             ("connected", 0.5, False),
                                             ("invalid", None, False)])
def test_classification_threshold_init(dummyexplainer, model,
                                       model_return_type, task_level,
                                       explanation_type, mask_type,
                                       threshold_pairs):
    if threshold_pairs[2]:
        Explainer(explanation_algorithm=dummyexplainer, model=model,
                  model_return_type=model_return_type, task_level=task_level,
                  explanation_type=explanation_type, mask_type=mask_type,
                  threshold=threshold_pairs[0],
                  threshold_value=threshold_pairs[1])

    else:
        with pytest.raises(ValueError):
            Explainer(explanation_algorithm=dummyexplainer, model=model,
                      model_return_type=model_return_type,
                      task_level=task_level, explanation_type=explanation_type,
                      mask_type=mask_type, threshold=threshold_pairs[0],
                      threshold_value=threshold_pairs[1])


@pytest.mark.parametrize("model", models[:1])
@pytest.mark.parametrize("model_return_type", model_return_types[:1])
@pytest.mark.parametrize("task_level", task_levels[:1])
@pytest.mark.parametrize("explanation_type", explanation_types + ["invalid"])
@pytest.mark.parametrize("mask_type", mask_types + ["invalid"])
@pytest.mark.parametrize("threshold", thresholds)
def test_classification_explainer_init(dummyexplainer, model,
                                       model_return_type, task_level,
                                       explanation_type, mask_type, threshold):
    if explanation_type == "invalid" or mask_type == "invalid":
        with pytest.raises(ValueError):
            Explainer(explanation_algorithm=dummyexplainer, model=model,
                      model_return_type=model_return_type,
                      task_level=task_level, explanation_type=explanation_type,
                      mask_type=mask_type, threshold=threshold)
    else:
        Explainer(explanation_algorithm=dummyexplainer, model=model,
                  model_return_type=model_return_type, task_level=task_level,
                  explanation_type=explanation_type, mask_type=mask_type,
                  threshold=threshold)


def test_init_model_not_compatible(dummyexplainer):
    with pytest.raises(ValueError):
        Explainer(explanation_algorithm=dummyexplainer,
                  model=DummyModelNotCompatible(),
                  model_return_type="regression", task_level="node",
                  explanation_type="phenomenon", mask_type="node",
                  threshold="none")


@pytest.mark.parametrize("model", models)
def test_get_prediction(dummyexplainer, dummyInput, model):
    explainer = Explainer(explanation_algorithm=dummyexplainer, model=model,
                          model_return_type="regression", task_level="node",
                          explanation_type="phenomenon", mask_type="node",
                          threshold="none")
    assert explainer.get_prediction(dummyInput).shape == (model.out_dim, )


@pytest.mark.parametrize("target", [None, torch.randn(2)])
@pytest.mark.parametrize("explanation_type", explanation_types)
def test_forward_target_switch(dummyInput, target, explanation_type):
    explainer = Explainer(explanation_algorithm=RandomExplainer(),
                          model=DummyModel(), model_return_type="regression",
                          task_level="node", explanation_type=explanation_type,
                          mask_type="node_feat", threshold="none")
    if target is None and explanation_type == "phenomenon":
        with pytest.raises(ValueError):
            explainer.forward(dummyInput, target=target)
    else:
        assert explainer.forward(
            dummyInput, target).node_features_mask.shape == dummyInput.x.shape


@pytest.mark.parametrize("threshold_value", [0.2, 0.5, 0.8])
def test_hard_threshold(dummyInput, threshold_value):
    explainer = Explainer(explanation_algorithm=RandomExplainer(),
                          model=DummyModel(), model_return_type="regression",
                          task_level="graph", explanation_type="phenomenon",
                          mask_type="node_feat", threshold="hard",
                          threshold_value=threshold_value)
    explanations_raw = explainer._compute_explanation(dummyInput, target=None,
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
