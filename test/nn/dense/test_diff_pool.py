import torch

from torch_geometric.nn import dense_diff_pool
from torch_geometric.testing import is_full_test


def test_dense_diff_pool():
    batch_size, num_nodes, channels, num_clusters = (2, 20, 16, 10)
    x = torch.randn((batch_size, num_nodes, channels))
    adj = torch.rand((batch_size, num_nodes, num_nodes))
    s = torch.randn((batch_size, num_nodes, num_clusters))
    mask = torch.randint(0, 2, (batch_size, num_nodes), dtype=torch.bool)

    x_out, adj_out, link_loss, ent_loss = dense_diff_pool(x, adj, s, mask)
    assert x_out.size() == (2, 10, 16)
    assert adj_out.size() == (2, 10, 10)
    assert link_loss.item() >= 0
    assert ent_loss.item() >= 0

    if is_full_test():
        jit = torch.jit.script(dense_diff_pool)
        x_jit, adj_jit, link_loss, ent_loss = jit(x, adj, s, mask)
        assert torch.allclose(x_jit, x_out)
        assert torch.allclose(adj_jit, adj_out)
        assert link_loss.item() >= 0
        assert ent_loss.item() >= 0
