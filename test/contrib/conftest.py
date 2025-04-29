import pytest
import torch

from torch_geometric.contrib.nn.layers.transformer import (
    GraphTransformerEncoderLayer,
)
from torch_geometric.contrib.nn.models import GraphTransformer
from torch_geometric.contrib.nn.positional.degree import DegreeEncoder
from torch_geometric.contrib.nn.positional.eigen import EigEncoder
from torch_geometric.contrib.nn.positional.svd import SVDEncoder
from torch_geometric.data import Batch, Data

# ── Basic graph‐batch fixtures ──────────────────────────────────────────────


@pytest.fixture
def two_graph_batch():
    """Creates a batch containing two identical graphs for testing.

    Each graph contains 3 nodes with 8-dimensional random features and no edges

    Returns:
        Batch: A batch object containing two graphs with identical structure.
    """
    data_list = []
    for _ in range(2):
        x = torch.randn(3, 8)
        data_list.append(
            Data(x=x, edge_index=torch.empty((2, 0), dtype=torch.long))
        )
    return Batch.from_data_list(data_list)


@pytest.fixture
def edge_batch():
    """Creates batched graph data with edge distance information.

    Factory function that generates graphs with random node features and edge
    distance matrices.

    Args:
        batch_size (int): Number of graphs in the batch
        seq_len (int): Number of nodes per graph
        num_edges (int): Maximum value for edge distance
        feat_dim (int, optional): Node feature dimension. Defaults to 1

    Returns:
        Callable: Function that produces a Batch with edge_dist attribute
    """

    def make(batch_size, seq_len, num_edges, feat_dim=1):
        data_list = []
        for _ in range(batch_size):
            edge_dist = torch.randint(0, num_edges, (seq_len, seq_len))
            x = torch.randn(seq_len, feat_dim)
            data_list.append(
                Data(
                    x=x,
                    edge_dist=edge_dist,
                    edge_index=torch.empty((2, 0), dtype=torch.long),
                )
            )
        return Batch.from_data_list(data_list)

    return make


@pytest.fixture
def hop_batch():
    """Creates batched graph data with hop distance information.

    Factory function that generates graphs with random node features and hop
    distance matrices.

    Args:
        batch_size (int): Number of graphs in the batch
        seq_len (int): Number of nodes per graph
        num_hops (int): Maximum hop distance value
        feat_dim (int, optional): Node feature dimension. Defaults to 1

    Returns:
        Callable: Function that produces a Batch with hop_dist attribute
    """

    def make(batch_size, seq_len, num_hops, feat_dim=1):
        data_list = []
        for _ in range(batch_size):
            hop_dist = torch.randint(0, num_hops, (seq_len, seq_len))
            x = torch.randn(seq_len, feat_dim)
            data_list.append(
                Data(
                    x=x,
                    hop_dist=hop_dist,
                    edge_index=torch.empty((2, 0), dtype=torch.long),
                )
            )
        return Batch.from_data_list(data_list)

    return make


@pytest.fixture
def spatial_batch():
    """Creates batched graph data with spatial position information.

    Factory function that generates graphs with random node features and
    spatial position matrices.

    Args:
        batch_size (int): Number of graphs in the batch
        seq_len (int): Number of nodes per graph
        num_spatial (int): Maximum spatial position value
        feat_dim (int, optional): Node feature dimension. Defaults to 1

    Returns:
        Callable: Function that produces a Batch with spatial_pos attribute
    """

    def make(batch_size, seq_len, num_spatial, feat_dim=1):
        data_list = []
        for _ in range(batch_size):
            spatial_pos = torch.randint(0, num_spatial, (seq_len, seq_len))
            x = torch.randn(seq_len, feat_dim)
            data_list.append(
                Data(
                    x=x,
                    spatial_pos=spatial_pos,
                    edge_index=torch.empty((2, 0), dtype=torch.long),
                )
            )
        return Batch.from_data_list(data_list)

    return make


@pytest.fixture
def spatial_edge_batch():
    """Creates batched graphs with both spatial and edge distance information.

    Factory function that generates graphs with random node features, spatial
    positions, and edge distances.

    Args:
        batch_size (int): Number of graphs in the batch
        seq_len (int): Number of nodes per graph
        num_spatial (int): Maximum spatial position value
        num_edges (int): Maximum edge distance value
        feat_dim (int): Node feature dimension

    Returns:
        Callable: Function that produces a Batch with spatial_pos and edge_dist
    """

    def make(batch_size, seq_len, num_spatial, num_edges, feat_dim):
        data_list = []
        for _ in range(batch_size):
            spatial_pos = torch.randint(0, num_spatial, (seq_len, seq_len))
            edge_dist = torch.randint(0, num_edges, (seq_len, seq_len))
            x = torch.randn(seq_len, feat_dim)
            data_list.append(
                Data(
                    x=x,
                    spatial_pos=spatial_pos,
                    edge_dist=edge_dist,
                    edge_index=torch.empty((2, 0), dtype=torch.long),
                )
            )
        return Batch.from_data_list(data_list)

    return make


# ── “Simple” fully‐connected graph batch ────────────────────────────────────


@pytest.fixture
def simple_batch():
    """Creates a single fully-connected graph batch.

    Args:
        feat_dim (int): Node feature dimension
        num_nodes (int): Number of nodes in the graph

    Returns:
        Callable: Function that produces a Batch with a fully-connected graph
    """

    def make(feat_dim, num_nodes):
        x = torch.randn(num_nodes, feat_dim)
        edge_index = torch.combinations(torch.arange(num_nodes), r=2).t()
        return Batch.from_data_list([Data(x=x, edge_index=edge_index)])

    return make


# ── Fixtures for bias‐provider tests ────────────────────────────────────────


@pytest.fixture
def dummy_provider():
    """Creates a simple bias provider for testing.

    Returns a module with a single learnable scale parameter that transforms
    input batches into attention bias tensors.

    Returns:
        nn.Module: Bias provider with shape (B, 4, N, N) output
    """

    class Dummy(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.ones(1))

        def forward(self, batch):
            B = int(batch.batch.max()) + 1
            N = int(torch.bincount(batch.batch).max())
            return torch.ones(B, 4, N, N) * 3.14 * self.scale

    return Dummy()


# ── Positional encoder classes (for GraphTransformer tests) ──────────────────


@pytest.fixture
def svd_encoder():
    """Provides the SVDEncoder class for positional encoding."""
    return SVDEncoder


@pytest.fixture
def eig_encoder():
    """Provides the EigEncoder class for positional encoding."""
    return EigEncoder


@pytest.fixture
def degree_encoder():
    """Provides the DegreeEncoder class for positional encoding."""
    return DegreeEncoder


# ── GraphTransformer factory ─────────────────────────────────────────────────


@pytest.fixture
def transformer_model():
    """Creates a GraphTransformer model factory.

    Returns:
        Callable: Function that instantiates a GraphTransformer with given
        kwargs
    """

    def make(**kwargs):
        return GraphTransformer(**kwargs)

    return make


# ── Transformer‐layer tests fixtures ─────────────────────────────────────────


@pytest.fixture(params=[(32, 4, 0.0)])
def encoder_layer(request):
    """Creates a GraphTransformerEncoderLayer for testing.

    Args from params:
        hidden_dim (int): Hidden dimension size, default 32
        num_heads (int): Number of attention heads, default 4
        dropout (float): Dropout probability, default 0.0

    Returns:
        GraphTransformerEncoderLayer: Configured transformer encoder layer
    """
    hidden_dim, num_heads, dropout = request.param
    return GraphTransformerEncoderLayer(
        hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout
    )


@pytest.fixture(params=[(2, 10), (1, 20)])
def input_batch(request):
    """Creates input tensors for transformer layer testing.

    Args from params:
        batch_size (int): Number of graphs, either 1 or 2
        num_nodes (int): Nodes per graph, either 10 or 20

    Returns:
        tuple: (x, batch) tensors where:
            - x: Node features of shape (batch_size * num_nodes, hidden_dim)
            - batch: Graph indices of shape (batch_size * num_nodes,)
    """
    batch_size, num_nodes = request.param
    hidden_dim = 32
    x = torch.randn(batch_size * num_nodes, hidden_dim)
    batch = torch.arange(batch_size).repeat_interleave(num_nodes)
    return x, batch


# ── Perf tests factories ─────────────────────────────────────────────────────


@pytest.fixture
def make_perf_batch():
    """Creates performance testing batch data.

    Args:
        num_graphs (int): Number of graphs in batch
        nodes_per_graph (int): Number of nodes per graph
        feat_dim (int): Node feature dimension
        device (torch.device): Device to create tensors on

    Returns:
        Callable: Function that produces a Batch for performance testing
    """

    def make(num_graphs, nodes_per_graph, feat_dim, device):
        data_list = []
        for _ in range(num_graphs):
            x = torch.randn(nodes_per_graph, feat_dim, device=device)
            data_list.append(
                Data(
                    x=x,
                    edge_index=torch.empty(
                        (2, 0), dtype=torch.long, device=device
                    )
                )
            )
        return Batch.from_data_list(data_list)

    return make


@pytest.fixture
def make_perf_batch_with_spatial():
    """Creates performance testing batch data with spatial positions.

    Similar to make_perf_batch but includes spatial position information.

    Args:
        num_graphs (int): Number of graphs in batch
        nodes_per_graph (int): Number of nodes per graph
        feat_dim (int): Node feature dimension
        num_spatial (int): Maximum spatial position value
        device (torch.device): Device to create tensors on

    Returns:
        Callable: Function that produces a Batch with spatial positions
    """

    def make(num_graphs, nodes_per_graph, feat_dim, num_spatial, device):
        data_list = []
        for _ in range(num_graphs):
            x = torch.randn(nodes_per_graph, feat_dim, device=device)
            spatial_pos = torch.randint(
                0,
                num_spatial, (nodes_per_graph, nodes_per_graph),
                device=device
            )
            data_list.append(
                Data(
                    x=x,
                    edge_index=torch.empty(
                        (2, 0), dtype=torch.long, device=device
                    ),
                    spatial_pos=spatial_pos
                )
            )
        return Batch.from_data_list(data_list)

    return make
