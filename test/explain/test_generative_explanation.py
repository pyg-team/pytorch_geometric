import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.explain import (Explainer, XGNNExplainer,
                                     GenerativeExplanation,
                                     ExplanationSetSampler)


# Mock model for testing
class MLP_Graph(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP_Graph, self).__init__()
        self.fc1 = nn.Linear(input_dim, 8)
        self.fc2 = nn.Linear(8, output_dim)

    def forward(self, x):
        # Flatten the graph representation
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Mock explainer algorithm
class ExampleExplainer(XGNNExplainer):
    def __init__(self, epochs, lr, candidate_set, validity_args):
        super(ExampleExplainer, self).__init__()
        self.epochs = epochs
        self.lr = lr
        self.candidate_set = candidate_set
        self.validity_args = validity_args

    def train_generative_model(self, model_to_explain, for_class):
        # For simplicity, this example does not include actual training logic

        for epoch in range(self.epochs):
            # Placeholder for training logic
            pass

        return Data()


# Mock graph generator
class ExampleGraphGenerator(torch.nn.Module, ExplanationSetSampler):
    def __init__(self, graph):
        self.graph = graph

    def sample(self):
        # has to return a list of Data objects
        return [Data(), Data(), Data()]


# Fixture for setting up XGNNExplainer
@pytest.fixture
def setup_xgnn_explainer():
    mock_model = MLP_Graph(input_dim=7, output_dim=1)

    explainer = Explainer(
        model=mock_model,
        algorithm=ExampleExplainer(
            epochs=10,
            lr=0.01,
            candidate_set={'C': torch.tensor([1, 0, 0, 0, 0, 0,
                                              0])},  # Simplified candidate set
            validity_args={'C': 4}),
        explanation_type='generative',
        model_config=dict(
            mode='binary_classification',
            task_level='graph',
            return_type='probs',
        ))

    class_index = 1
    x = torch.tensor([])
    edge_index = torch.tensor([[], []])

    return explainer, x, edge_index, class_index


# Test output of XGNNExplainer
def test_explainer_output(setup_xgnn_explainer):
    explainer, x, edge_index, class_index = setup_xgnn_explainer
    explanation = explainer(x, edge_index, for_class=class_index)

    # Check if explanation is of type Data
    assert isinstance(explanation, Data), "Explanation is not of type Data"


# Test output of ExampleExplainer
def test_sampler_output():
    sampled_graphs = ExampleGraphGenerator(Data()).sample()

    # Check if sampled_graphs is a list of Data objects
    assert isinstance(sampled_graphs, list), "Sampled graphs is not a list"
    assert all(isinstance(graph, Data) for graph in
               sampled_graphs), "Sampled graphs is not a list of Data objects"


# Test is_finite method
def test_is_finite():
    gen_explanation = GenerativeExplanation(
        explanation_set=ExampleGraphGenerator, is_finite=True)
    assert gen_explanation.is_finite() is True

    gen_explanation = GenerativeExplanation(
        explanation_set=ExampleGraphGenerator, is_finite=False)
    assert gen_explanation.is_finite() is False


# Test get_explanation_set method
def test_get_explanation_set():
    gen_explanation = GenerativeExplanation(
        explanation_set=ExampleGraphGenerator(Data()), is_finite=True)
    assert isinstance(gen_explanation.get_explanation_set(), list)
