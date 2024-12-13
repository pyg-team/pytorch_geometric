import numpy as np
import pytest
import torch

from torch_geometric.contrib.nn import TokenGT
from torch_geometric.data import Batch, Data


def create_batch(num_nodes, num_edges, input_feat_dim=16):
    """Generates a random batch of graphs to test TokenGT on.
    In each graph, edges are generated randomly between nodes.

    Args:
        - input_feat_dim: dimension of the node and edge features
        - num_nodes: list where element i denotes number of nodes in graph i
        - num_edges: list where element i denotes number of edges in graph i
    """
    all_data = []
    for n_nodes, n_edges in zip(num_nodes, num_edges):
        x = torch.randn(n_nodes, input_feat_dim)
        edge_index = torch.tensor(
            np.random.choice(range(n_nodes), size=(2, n_edges)))
        edge_attr = torch.randn(n_edges, input_feat_dim)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.batch = torch.zeros(n_nodes, dtype=torch.long)
        all_data.append(data)
    batch = Batch.from_data_list(all_data)
    return batch


def run_token_gt(batch, input_feat_dim=16, hidden_dim=32, num_layers=2,
                 num_heads=4, num_classes=10, d_p=9, d_e=4, method="orf"):
    model = TokenGT(input_feat_dim=input_feat_dim, hidden_dim=hidden_dim,
                    num_layers=num_layers, num_heads=num_heads,
                    num_classes=num_classes, method=method, d_p=d_p, d_e=d_e,
                    use_graph_token=True)
    logits = model(batch)
    return logits


@pytest.mark.parametrize("input_feat_dim", range(16))
def test_token_gt_orf(input_feat_dim):
    num_classes = 10
    batch = create_batch(num_nodes=[5], num_edges=[4],
                         input_feat_dim=input_feat_dim)
    logits = run_token_gt(batch, input_feat_dim=input_feat_dim,
                          num_classes=num_classes, method="orf")
    assert logits.size() == (1, num_classes)
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("input_feat_dim", range(16))
def test_token_gt_laplacian(input_feat_dim):
    num_classes = 10
    batch = create_batch(num_nodes=[5], num_edges=[4],
                         input_feat_dim=input_feat_dim)
    logits = run_token_gt(batch, input_feat_dim=input_feat_dim,
                          num_classes=num_classes, method="laplacian")
    assert logits.size() == (1, num_classes)
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("method", ["orf", "laplacian"])
def test_token_gt_batches(method):
    input_feat_dim, num_classes, num_graphs = 16, 10, 8
    num_nodes = np.random.choice(range(1, 20), size=num_graphs)
    num_edges = [int(np.random.random() * n * 2) for n in num_nodes]
    batch = create_batch(num_nodes=num_nodes, num_edges=num_edges,
                         input_feat_dim=input_feat_dim)
    logits = run_token_gt(batch, input_feat_dim=input_feat_dim,
                          num_classes=num_classes, method=method)
    assert logits.size() == (num_graphs, num_classes)
    assert torch.isfinite(logits).all()
