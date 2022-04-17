import torch

from torch_geometric.nn import dense_mincut_pool


def test_dense_mincut_pool():
    batch_size, num_nodes, channels, num_clusters = (2, 20, 16, 10)
    x = torch.randn((batch_size, num_nodes, channels))
    adj = torch.ones((batch_size, num_nodes, num_nodes))
    s = torch.randn((batch_size, num_nodes, num_clusters))
    mask = torch.randint(0, 2, (batch_size, num_nodes), dtype=torch.bool)

    x, adj, mincut_loss, ortho_loss = dense_mincut_pool(x, adj, s, mask)
    assert x.size() == (2, 10, 16)
    assert adj.size() == (2, 10, 10)
    assert -1 <= mincut_loss <= 0
    assert 0 <= ortho_loss <= 2
