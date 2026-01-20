import pytest
import torch

from torch_geometric.nn.attention.expander import ExpanderAttention
from torch_geometric.nn.attention.local import LocalAttention
from torch_geometric.transforms.exphormer import EXPHORMER


def create_mock_data(num_nodes, hidden_dim, edge_dim=None):
    from torch_geometric.data import Data
    x = torch.rand(num_nodes, hidden_dim)
    edge_index = torch.randint(0, num_nodes,
                               (2, num_nodes * 2))  # Create a few random edges

    # Ensure edge_attr has the same number of rows as edges in edge_index
    edge_attr = torch.rand(edge_index.size(1), edge_dim) if edge_dim else None

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# Tests for LocalAttention
@pytest.fixture
def local_attention_model():
    hidden_dim, num_heads = 16, 4
    return LocalAttention(hidden_dim=hidden_dim, num_heads=num_heads)


def test_local_attention_initialization(local_attention_model):
    assert local_attention_model.hidden_dim == 16
    assert local_attention_model.num_heads == 4
    assert local_attention_model.q_proj.weight.shape == (16, 16)


def test_local_attention_forward(local_attention_model):
    data = create_mock_data(num_nodes=10, hidden_dim=16, edge_dim=16)
    output = local_attention_model(data.x, data.edge_index, data.edge_attr)
    assert output.shape == (data.x.shape[0], 16)


def test_local_attention_no_edge_attr(local_attention_model):
    data = create_mock_data(num_nodes=10, hidden_dim=16, edge_dim=16)
    output_no_edge_attr = local_attention_model(data.x, data.edge_index, None)
    assert output_no_edge_attr.shape == (data.x.shape[0], 16)


def test_local_attention_mismatched_edge_attr(local_attention_model):
    data = create_mock_data(num_nodes=10, hidden_dim=16, edge_dim=16)
    mismatched_edge_attr = torch.rand(data.edge_attr.size(0) - 1, 16)
    with pytest.raises(ValueError, match="edge_attr size doesn't match"):
        local_attention_model(data.x, data.edge_index, mismatched_edge_attr)


def test_local_attention_message(local_attention_model):
    q_i = torch.rand(20, 4, 4)
    k_j = torch.rand_like(q_i)
    v_j = torch.rand_like(q_i)
    edge_attr = torch.rand(20, 4)
    attention_scores = local_attention_model.message(q_i, k_j, v_j, edge_attr)
    assert attention_scores.shape == v_j.shape


# Tests for ExpanderAttention
@pytest.fixture
def expander_attention_model():
    hidden_dim, num_heads, expander_degree = 16, 4, 4
    return ExpanderAttention(hidden_dim=hidden_dim,
                             expander_degree=expander_degree,
                             num_heads=num_heads)


def test_expander_attention_initialization(expander_attention_model):
    assert expander_attention_model.expander_degree == 4
    assert expander_attention_model.q_proj.weight.shape == (16, 16)


def test_expander_attention_generate_expander_edges(expander_attention_model):
    edge_index = expander_attention_model.generate_expander_edges(num_nodes=10)
    assert edge_index.shape[0] == 2


def test_expander_attention_forward(expander_attention_model):
    data = create_mock_data(num_nodes=10, hidden_dim=16)
    output, edge_index = expander_attention_model(data.x, num_nodes=10)
    assert output.shape == (data.x.shape[0], 16)


def test_expander_attention_insufficient_nodes(expander_attention_model):
    with pytest.raises(Exception):
        expander_attention_model.generate_expander_edges(2)


def test_expander_attention_message(expander_attention_model):
    q_i = torch.rand(20, 4, 4)
    k_j = torch.rand_like(q_i)
    v_j = torch.rand_like(q_i)
    attention_scores = expander_attention_model.message(q_i, k_j, v_j)
    assert attention_scores.shape == v_j.shape


# Tests for EXPHORMER
@pytest.fixture
def exphormer_model():
    hidden_dim, num_layers, num_heads, expander_degree = 16, 3, 4, 4
    return EXPHORMER(hidden_dim=hidden_dim, num_layers=num_layers,
                     num_heads=num_heads, expander_degree=expander_degree)


def test_exphormer_initialization(exphormer_model):
    assert len(exphormer_model.layers) == 3
    assert isinstance(exphormer_model.layers[0]['local'], LocalAttention)
    if exphormer_model.use_expander:
        assert isinstance(exphormer_model.layers[0]['expander'],
                          ExpanderAttention)


def test_exphormer_forward(exphormer_model):
    data = create_mock_data(num_nodes=10, hidden_dim=16)
    output = exphormer_model(data)
    assert output.shape == (data.x.shape[0], 16)


def test_exphormer_empty_graph(exphormer_model):
    data_empty = create_mock_data(num_nodes=1, hidden_dim=16)
    with pytest.raises(Exception):
        exphormer_model(data_empty)


def test_exphormer_incorrect_input_dimensions(exphormer_model):
    data_incorrect_dim = create_mock_data(num_nodes=10, hidden_dim=17)
    with pytest.raises(Exception):
        exphormer_model(data_incorrect_dim)
