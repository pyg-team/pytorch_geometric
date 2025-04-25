# test_graph_transformer.py

import pytest
import torch

from torch_geometric.contrib.nn.models import GraphTransformer
from torch_geometric.data import Batch, Data


@pytest.mark.parametrize(
    "num_graphs, num_nodes, feature_dim, num_classes",
    [
        (1, 10, 16, 2),  # Single graph with 10 nodes
        (5, 20, 32, 3),  # Multiple graphs with different node counts
        (3, 15, 64, 4),  # Multiple graphs with same node count
    ],
    ids=[
        "single_graph", "multiple_graphs_diff_nodes",
        "multiple_graphs_same_nodes"
    ])
def test_forward_shape(num_graphs, num_nodes, feature_dim, num_classes):
    data_list = []
    for _ in range(num_graphs):
        x = torch.randn(num_nodes, feature_dim)
        edge_index = torch.empty((2, 0), dtype=torch.long)  # No edges
        data_list.append(Data(x=x, edge_index=edge_index))
    batch = Batch.from_data_list(data_list)
    model = GraphTransformer(hidden_dim=feature_dim, num_class=num_classes)
    out = model(batch)["logits"]
    assert out.shape == (
        num_graphs, num_classes
    ), f"Expected output shape {(num_graphs, num_classes)}, got {out.shape}"
