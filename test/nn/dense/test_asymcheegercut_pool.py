import torch

from torch_geometric.nn import dense_asymcheegercut_pool
from torch_geometric.testing import is_full_test


def test_dense_asymcheegercut_pool():
    batch_size, num_nodes, channels, num_clusters = (2, 20, 16, 10)
    x = torch.randn((batch_size, num_nodes, channels))
    adj = torch.ones((batch_size, num_nodes, num_nodes))
    s = torch.randn((batch_size, num_nodes, num_clusters))
    mask = torch.randint(0, 2, (batch_size, num_nodes), dtype=torch.bool)

    x_out, adj_out, totvar_loss, balance_loss = dense_asymcheegercut_pool(
        x, adj, s, mask)
    assert x_out.size() == (2, 10, 16)
    assert adj_out.size() == (2, 10, 10)
    assert 0 <= totvar_loss <= 1
    assert 0 <= balance_loss <= 1

    if is_full_test():
        jit = torch.jit.script(dense_asymcheegercut_pool)

        x_jit, adj_jit, totvar_loss, balance_loss = jit(x, adj, s, mask)
        assert x_jit.size() == (2, 10, 16)
        assert adj_jit.size() == (2, 10, 10)
        assert 0 <= totvar_loss <= 1
        assert 0 <= balance_loss <= 1
