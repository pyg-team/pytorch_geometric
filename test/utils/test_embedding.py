import pytest
import torch

from torch_geometric.nn import GCNConv, Linear
from torch_geometric.utils import get_embeddings
from torch_geometric.utils.embedding import get_embeddings_hetero


class GNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(5, 6)
        self.conv2 = GCNConv(6, 7)

    def forward(self, x0, edge_index):
        x1 = self.conv1(x0, edge_index)
        x2 = self.conv2(x1, edge_index)
        return [x1, x2]


def test_get_embeddings():
    x = torch.randn(6, 5)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])

    with pytest.warns(UserWarning, match="any 'MessagePassing' layers"):
        intermediate_outs = get_embeddings(Linear(5, 5), x)
    assert len(intermediate_outs) == 0

    model = GNN()
    expected_embeddings = model(x, edge_index)

    embeddings = get_embeddings(model, x, edge_index)
    assert len(embeddings) == 2
    for expected, out in zip(expected_embeddings, embeddings, strict=False):
        assert torch.allclose(expected, out)


def test_get_embeddings_hetero(hetero_data, hetero_model):
    # Create model using the metadata from hetero_data
    metadata = hetero_data.metadata()
    model = hetero_model(metadata)

    # Get heterogeneous embeddings
    embeddings_dict = get_embeddings_hetero(model, None, hetero_data.x_dict,
                                            hetero_data.edge_index_dict)

    # Verify the structure of the returned embeddings
    assert isinstance(embeddings_dict, dict)
    assert 'paper' in embeddings_dict
    assert 'author' in embeddings_dict

    # Verify that we have embeddings for both node types
    assert len(embeddings_dict['paper']) > 0
    assert len(embeddings_dict['author']) > 0

    # Check that the embeddings have the right shape
    num_paper_nodes = hetero_data['paper'].num_nodes
    num_author_nodes = hetero_data['author'].num_nodes

    # Verify dimensions of embeddings
    assert embeddings_dict['paper'][0].shape == (num_paper_nodes, 32
                                                 )  # First layer
    assert embeddings_dict['author'][0].shape == (num_author_nodes, 32
                                                  )  # First layer
