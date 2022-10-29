import pytest
import torch

from torch_geometric.data import Data
from torch_geometric.explainability.algo import RandomExplainer
from torch_geometric.explainability.explainer import (
    Explainer,
    ExplainerConfig,
    ModelConfig,
    Threshold,
)


class DummyModel(torch.nn.Module):
    def __init__(self, out_dim: int = 1) -> None:
        super().__init__()
        self.out_dim = out_dim

    def forward(self, g, **kwargs):
        return g.x.mean().repeat(self.out_dim)


@pytest.fixture
def dummyInput():
    return Data(x=torch.randn(10, 5), edge_index=torch.randint(0, 10, (2, 20)),
                edge_attr=torch.randn(20, 3), target=torch.randn(10, 1))


@pytest.fixture
def explanation_algorithm():
    return RandomExplainer()


@pytest.fixture
def loss():
    return torch.nn.MSELoss()


models = [DummyModel(), DummyModel(2)]
model_return_type = ModelConfig._valid_model_return_type
explanation_types = ExplainerConfig._valid_explanation_type
mask_types = [x for x in ExplainerConfig._valid_mask_type if x != "layers"]
thresholds = [x for x in Threshold._valid_type if x != "connected"]

# test initialization


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("model_return_type", model_return_type)
@pytest.mark.parametrize("explanation_type", [explanation_types[0]])
@pytest.mark.parametrize("mask_type", mask_types)
@pytest.mark.parametrize("threshold", thresholds + ["connected"])
def test_explainer_threshold_init(explanation_type, explanation_algorithm,
                                  mask_type, threshold, loss, model,
                                  model_return_type):
    threshold_values = [0.5, 0.7, 0.9]
    if threshold == "none":
        valid_values = threshold_values
        invalid_values = ["abc"]
    elif threshold == "hard":
        valid_values = threshold_values
        invalid_values = ["abc", 12, -1]
    elif threshold == "topk":
        valid_values = [1, 3, 6]
        invalid_values = ["abc", -1] + threshold_values

    if threshold == "connected":
        with pytest.raises(NotImplementedError):
            Explainer(explanation_type=explanation_type,
                      explanation_algorithm=explanation_algorithm,
                      mask_type=mask_type, threshold=threshold, loss=loss,
                      model=model, model_return_type=model_return_type,
                      threshold_value=12)
    else:
        for value in valid_values:
            Explainer(explanation_type=explanation_type,
                      explanation_algorithm=explanation_algorithm,
                      mask_type=mask_type, threshold=threshold, loss=loss,
                      model=model, model_return_type=model_return_type,
                      threshold_value=value)
        for value in invalid_values:
            with pytest.raises(ValueError):
                Explainer(explanation_type=explanation_type,
                          explanation_algorithm=explanation_algorithm,
                          mask_type=mask_type, threshold=threshold, loss=loss,
                          model=model, model_return_type=model_return_type,
                          threshold_value=value)


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("model_return_type", model_return_type)
@pytest.mark.parametrize("explanation_type", [explanation_types[0]])
@pytest.mark.parametrize("mask_type", mask_types + ["unseen"])
@pytest.mark.parametrize("threshold", thresholds)
def test_explainer_mask_type_init(explanation_type, explanation_algorithm,
                                  mask_type, threshold, loss, model,
                                  model_return_type):
    if mask_type == "unseen":
        with pytest.raises(ValueError):
            Explainer(explanation_type=explanation_type,
                      explanation_algorithm=explanation_algorithm,
                      mask_type=mask_type, threshold=threshold, loss=loss,
                      model=model, model_return_type=model_return_type,
                      threshold_value=1)
    else:
        Explainer(explanation_type=explanation_type,
                  explanation_algorithm=explanation_algorithm,
                  mask_type=mask_type, threshold=threshold, loss=loss,
                  model=model, model_return_type=model_return_type,
                  threshold_value=1)


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("model_return_type",
                         model_return_type + ["raw", "invalid"])
@pytest.mark.parametrize("explanation_type", [explanation_types[0]])
@pytest.mark.parametrize("mask_type", mask_types)
@pytest.mark.parametrize("threshold", thresholds)
def test_explainer_model_return_init(explanation_type, explanation_algorithm,
                                     mask_type, threshold, loss, model,
                                     model_return_type):
    if model_return_type in ["raw", "invalid"]:
        with pytest.raises(ValueError):
            Explainer(explanation_type=explanation_type,
                      explanation_algorithm=explanation_algorithm,
                      mask_type=mask_type, threshold=threshold, loss=loss,
                      model=model, model_return_type=model_return_type,
                      threshold_value=1)
    else:
        Explainer(explanation_type=explanation_type,
                  explanation_algorithm=explanation_algorithm,
                  mask_type=mask_type, threshold=threshold, loss=loss,
                  model=model, model_return_type=model_return_type,
                  threshold_value=1)


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("model_return_type", ["regression"])
@pytest.mark.parametrize("explanation_type", explanation_types)
@pytest.mark.parametrize("mask_type", ["node"])
@pytest.mark.parametrize("threshold", ["none"])
@pytest.mark.parametrize("threshold_value", [0.5])
def test_objective_func_init(explanation_type, explanation_algorithm,
                             mask_type, threshold, loss, model,
                             model_return_type, dummyInput, threshold_value):
    explainer = Explainer(explanation_type=explanation_type,
                          explanation_algorithm=explanation_algorithm,
                          mask_type=mask_type, threshold=threshold, loss=loss,
                          model=model, model_return_type=model_return_type,
                          threshold_value=threshold_value)
    out = model(dummyInput)
    dummy_explainer_output = torch.ones_like(out)
    target = torch.zeros_like(out)
    loss_result = explainer.explanation_algorithm.objective(
        dummy_explainer_output, out, target, 0)
    if explanation_type == "model":
        assert loss_result == loss(dummy_explainer_output, out)
    elif explanation_type == "phenomenon":
        assert loss_result == loss(dummy_explainer_output, target)


# test forward use


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("model_return_type", model_return_type)
@pytest.mark.parametrize("explanation_type", [explanation_types[0]])
@pytest.mark.parametrize("mask_type", mask_types)
@pytest.mark.parametrize("threshold", ["hard", "none"])
@pytest.mark.parametrize("threshold_value", [0, 0.2, 0.5, 1])
def test_forward_random_explainer(explanation_type, explanation_algorithm,
                                  mask_type, threshold, loss, model,
                                  model_return_type, dummyInput,
                                  threshold_value):
    explainer = Explainer(explanation_type=explanation_type,
                          explanation_algorithm=explanation_algorithm,
                          mask_type=mask_type, threshold=threshold, loss=loss,
                          model=model, model_return_type=model_return_type,
                          threshold_value=threshold_value)
    explanations = explainer.forward(g=dummyInput,
                                     y=dummyInput.target[0:model.out_dim])

    # Check size of explanations
    assert explanations.node_features_mask.size() == dummyInput.x.size()
    assert explanations.node_mask.size(dim=0) == dummyInput.x.size(dim=0)
    assert explanations.edge_mask.size(dim=0) == dummyInput.edge_index.size(
        dim=1)
    assert explanations.edge_features_mask.size() == dummyInput.edge_attr.size(
    )


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("model_return_type", model_return_type)
@pytest.mark.parametrize("explanation_type", [explanation_types[0]])
@pytest.mark.parametrize("mask_type", mask_types)
@pytest.mark.parametrize("threshold", ["hard"])
@pytest.mark.parametrize("threshold_value", [0.5])
def test_hard_tresholding_values(explanation_type, explanation_algorithm,
                                 mask_type, threshold, loss, model,
                                 model_return_type, dummyInput,
                                 threshold_value):
    explainer = Explainer(explanation_type=explanation_type,
                          explanation_algorithm=explanation_algorithm,
                          mask_type=mask_type, threshold=threshold, loss=loss,
                          model=model, model_return_type=model_return_type,
                          threshold_value=threshold_value)
    explanations_raw = explainer._compute_explanation(
        g=dummyInput, target=dummyInput.target[0:model.out_dim],
        target_index=0, batch=None)
    explanations_processed = explainer._threshold(explanations_raw)
    # check correct range of values before and after thresholding
    for exp in explanations_raw.available_explanations:
        assert torch.all(explanations_raw[exp] >= 0)
        assert torch.all(explanations_raw[exp] <= 1)
        assert torch.any((explanations_processed[exp] == 0)
                         | (explanations_processed[exp] == 1))
