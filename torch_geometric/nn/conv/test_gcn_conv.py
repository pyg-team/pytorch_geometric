import torch

from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv
from torch_geometric.typing import SparseTensor


def test_gcnconv_sparse_tensor_with_batching():
    torch.manual_seed(123)
    in_channels = 4
    out_channels = 8

    # Create two graphs (Graph 1 has 3 nodes, Graph 2 has 2 nodes)
    x1 = torch.randn(3, in_channels)
    edge_index1 = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 2]])
    edge_weight1 = torch.tensor([0.5, 1.0, 0.7, 0.9])

    x2 = torch.randn(2, in_channels)
    edge_index2 = torch.tensor([[0, 1], [1, 0]])
    edge_weight2 = torch.tensor([1.0, 0.8])

    data1 = Data(x=x1, edge_index=edge_index1, edge_weight=edge_weight1)
    data2 = Data(x=x2, edge_index=edge_index2, edge_weight=edge_weight2)

    # Batch them together
    batch = Batch.from_data_list([data1, data2])
    x = batch.x
    num_nodes = x.size(0)

    # Combine edge indices and weights
    edge_index = batch.edge_index
    edge_weight = batch.edge_weight

    # Create batched SparseTensor (global index is preserved)
    adj = SparseTensor.from_edge_index(edge_index, edge_weight,
                                       sparse_sizes=(num_nodes, num_nodes))

    # Apply GCNConv
    conv = GCNConv(in_channels, out_channels, cached=True)
    out1 = conv(x, adj)
    out2 = conv(x, adj)  # Reuse cache

    # Validate
    assert out1.shape == (num_nodes, out_channels)
    assert torch.allclose(out1, out2, atol=1e-6)

    # Optional: Check per-graph separation using batch vector
    batch_vector = batch.batch  # [0, 0, 0, 1, 1]
    assert batch_vector.size(0) == num_nodes
