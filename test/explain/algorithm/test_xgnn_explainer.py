import pytest
import torch
from torch_geometric.explain import GenerativeExplanation, XGNNExplainer


# Mock subclass of XGNNExplainer for testing
class MockXGNNExplainer(XGNNExplainer):
    def train_generative_model(self, model, for_class, **kwargs):
        return None


@pytest.fixture
def model():
    return torch.nn.Linear(3, 2)


def test_xgnn_explainer_initialization():
    explainer = MockXGNNExplainer(epochs=200, lr=0.005)
    assert explainer.epochs == 200
    assert explainer.lr == 0.005


def test_xgnn_explainer_forward(model):
    explainer = MockXGNNExplainer()
    x = torch.rand(10, 3)
    edge_index = torch.randint(0, 10, (2, 30))
    target = torch.randint(0, 2, (10, ))

    explanation = explainer(model, x, edge_index, target=target, for_class=1)
    assert isinstance(explanation, GenerativeExplanation)

    # Test ValueError for missing 'for_class' argument
    with pytest.raises(ValueError):
        explainer(model, x, edge_index, target=target)


def test_xgnn_explainer_abstract_method():
    class IncompleteExplainer(XGNNExplainer):
        pass

    explainer = IncompleteExplainer()

    # Ensure that instantiation fails due to the unimplemented abstract method
    with pytest.raises(NotImplementedError):
        explainer.train_generative_model(None, for_class=0)
