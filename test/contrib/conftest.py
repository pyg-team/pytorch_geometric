from typing import Callable, List, Sequence

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from torch_geometric.contrib.nn.bias import BaseBiasProvider
from torch_geometric.contrib.nn.layers.transformer import (
    GraphTransformerEncoderLayer,
)
from torch_geometric.contrib.nn.models import GraphTransformer
from torch_geometric.contrib.nn.positional.degree import DegreeEncoder
from torch_geometric.contrib.nn.positional.eigen import EigEncoder
from torch_geometric.contrib.nn.positional.svd import SVDEncoder
from torch_geometric.data import Batch, Data

# ── Benchmark “constants” ────────────────────────────────────────────────────


@pytest.fixture
def structural_config():
    """Parameters controlling graph‐structure (counts & feature sizes)."""
    return {
        "num_graphs": 32,
        "nodes_per_graph": 128,
        "feat_dim": 64,
    }


@pytest.fixture
def positional_config():
    """Parameters controlling positional & topological features."""
    return {
        "num_spatial": 16,
        "num_edges": 8,
        "num_hops": 4,
        "max_degree": 3,
        "num_eigvec": 4,
        "num_sing_vec": 3,
    }


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


@pytest.fixture
def simple_none_batch():
    """Creates a single graph batch with None node features.

    This reproduces the error scenario where datasets (like QM7B regression)
    don't provide node features, causing GraphTransformer to fail with:
    "linear(): argument 'input' (position 1) must be Tensor, not NoneType"

    Returns:
        Batch: A batch containing one graph with None node features
    """
    data = Data(
        x=None,  # No node features - this triggers the error
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
    )
    return Batch.from_data_list([data])


# ── Fixtures for bias‐provider tests ────────────────────────────────────────


@pytest.fixture
def dummy_provider():
    """Creates a simple bias provider for testing.

    Returns:
        DummyBias: An instance of a simple bias provider.
    """

    class DummyBias(BaseBiasProvider):

        def __init__(self):
            # 4 heads, no super-node
            super().__init__(num_heads=4, use_super_node=False)
            self.scale = nn.Parameter(torch.ones(1))

        def _extract_raw_distances(self, data):
            # Build a (B, N, N) zero‐distance tensor
            B = int(data.batch.max()) + 1
            N = int(torch.bincount(data.batch).max())
            return data.x.new_zeros((B, N, N), dtype=torch.long)

        def _embed_bias(self, distances):
            # distances: (B, N, N) → bias: (B, N, N, H)
            B, N, _ = distances.shape
            # constant 3.14 * scale across all entries and heads
            return (
                torch.ones(B, N, N, self.num_heads, device=distances.device) *
                3.14 * self.scale
            )

    return DummyBias()


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


# ── Full‐feature batch factory ───────────────────────────────────────────────


@pytest.fixture
def full_feature_batch(
    make_perf_batch_with_spatial, structural_config, positional_config
):
    """Builds a Batch that has *all* of:
      - spatial_pos, edge_dist, hop_dist
      - in_degree, out_degree
      - eig_pos_emb, svd_pos_emb
    using the two small config dicts above.
    """

    def make(device):
        # 1) start from a purely‐spatial batch
        batch = make_perf_batch_with_spatial(
            structural_config["num_graphs"],
            structural_config["nodes_per_graph"],
            structural_config["feat_dim"],
            positional_config["num_spatial"],
            device,
        )

        # 2) enrich each Data in the batch
        out_list = []
        for data in batch.to_data_list():
            N = data.x.size(0)
            # degrees
            data.in_degree = torch.randint(
                0, positional_config["max_degree"], (N, ), device=device
            )
            data.out_degree = torch.randint(
                0, positional_config["max_degree"], (N, ), device=device
            )
            # eigen & SVD positional embeddings
            data.eig_pos_emb = torch.randn(
                N, positional_config["num_eigvec"], device=device
            )
            data.svd_pos_emb = torch.randn(
                N, 2 * positional_config["num_sing_vec"], device=device
            )
            # edge & hop distances
            data.edge_dist = torch.randint(
                0, positional_config["num_edges"], (N, N), device=device
            )
            data.hop_dist = torch.randint(
                0, positional_config["num_hops"], (N, N), device=device
            )
            out_list.append(data)

        return Batch.from_data_list(out_list)

    return make


@pytest.fixture
def hetero_spatial_batch() -> Callable[[List[int], int, int], Data]:
    """Creates Data for heterogeneous graphs with padded spatial_pos.

    Factory function for Data objects with unequal-node graphs whose
    spatial_pos matrices are padded to a uniform size.

    Args:
        node_counts (List[int]): Number of nodes in each graph.
        feat_dim (int): Node feature dimensionality.
        num_spatial (int): Maximum spatial position value.

    Returns:
        Callable[[List[int], int, int], Data]: Function producing a Data
        object with padded spatial_pos and ptr attributes.
    """

    def make(node_counts: List[int], feat_dim: int, num_spatial: int) -> Data:
        max_n = max(node_counts)
        padded = []
        for n in node_counts:
            mat = torch.randint(0, num_spatial, (n, n))
            pad = (0, max_n - n, 0, max_n - n)
            padded.append(F.pad(mat, pad, value=0))
        spatial_pos = torch.cat(padded, dim=0)
        ptr = torch.tensor(
            [i * max_n for i in range(len(node_counts) + 1)],
            dtype=torch.long,
        )
        x = torch.randn(len(node_counts) * max_n, feat_dim)
        return Data(x=x, spatial_pos=spatial_pos, ptr=ptr)

    return make


@pytest.fixture
def blockdiag_spatial_batch() -> Data:
    """Create a Data object with block-diagonal spatial_pos for two graphs.

    Graph 1 has 11 nodes, Graph 2 has 14 nodes, so
    `spatial_pos` is a (25×25) block-diag matrix; `ptr=[0,11,25]`.
    """
    n1, n2 = 11, 14
    mat1 = torch.randint(0, 5, (n1, n1))
    mat2 = torch.randint(0, 5, (n2, n2))
    spatial_pos = torch.block_diag(mat1, mat2)
    ptr = torch.tensor([0, n1, n1 + n2], dtype=torch.long)
    x = torch.randn(n1 + n2, 8)
    return Data(x=x, spatial_pos=spatial_pos, ptr=ptr)


@pytest.fixture
def blockdiag_hop_batch() -> Callable[[Sequence[int], int, int], Data]:
    """Factory for Data with block-diagonal `hop_dist` and `ptr`.

    Each graph i has Ni nodes; we generate a random Ni×Ni hop_dist block,
    then assemble a big (∑Ni × ∑Ni) tensor that is block-diagonal,
    with zeros off the blocks.

    Returns:
        make(node_counts, feat_dim=1, num_hops=5) -> Data
    """

    def make(
        node_counts: Sequence[int],
        feat_dim: int = 1,
        num_hops: int = 5
    ) -> Data:
        # cumulative pointers [0, N1, N1+N2, …]
        ptr = [0]
        for n in node_counts:
            ptr.append(ptr[-1] + n)
        total = ptr[-1]

        # start with all-zero block-diagonal matrix
        hop_dist = torch.zeros((total, total), dtype=torch.long)
        for i, n in enumerate(node_counts):
            block = torch.randint(0, num_hops, (n, n), dtype=torch.long)
            s, e = ptr[i], ptr[i + 1]
            hop_dist[s:e, s:e] = block

        x = torch.randn(total, feat_dim)
        return Data(
            x=x,
            hop_dist=hop_dist,
            ptr=torch.tensor(ptr, dtype=torch.long),
            edge_index=torch.empty((2, 0), dtype=torch.long),
        )

    return make


@pytest.fixture
def blockdiag_edge_batch() -> Callable[[Sequence[int], int, int], Data]:
    """Factory for Data with block-diagonal `edge_dist` and `ptr`.

    Each graph i has Ni nodes; we generate a random Ni×Ni edge_dist block,
    then assemble a big (∑Ni × ∑Ni) tensor that is block-diagonal,
    with zeros elsewhere.

    Returns:
        make(node_counts, feat_dim=1, num_edges=5) -> Data
    """

    def make(
        node_counts: Sequence[int],
        feat_dim: int = 1,
        num_edges: int = 5
    ) -> Data:
        ptr = [0]
        for n in node_counts:
            ptr.append(ptr[-1] + n)
        total = ptr[-1]
        edge_dist = torch.zeros((total, total), dtype=torch.long)
        for i, n in enumerate(node_counts):
            block = torch.randint(0, num_edges, (n, n), dtype=torch.long)
            start, end = ptr[i], ptr[i + 1]
            edge_dist[start:end, start:end] = block

        x = torch.randn(total, feat_dim)
        return Data(
            x=x,
            edge_dist=edge_dist,
            ptr=torch.tensor(ptr, dtype=torch.long),
            edge_index=torch.empty((2, 0), dtype=torch.long),
        )

    return make
