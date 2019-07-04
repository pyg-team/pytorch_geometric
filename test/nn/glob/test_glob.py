import torch
from torch_geometric.nn import (global_add_pool, global_mean_pool,
                                global_max_pool)


def test_global_pool():
    N_1, N_2 = 4, 6
    x = torch.randn(N_1 + N_2, 4)
    batch = torch.tensor([0 for _ in range(N_1)] + [1 for _ in range(N_2)])

    out = global_add_pool(x, batch)
    assert out.size() == (2, 4)
    assert out[0].tolist() == x[:4].sum(dim=0).tolist()
    assert out[1].tolist() == x[4:].sum(dim=0).tolist()

    out = global_mean_pool(x, batch)
    assert out.size() == (2, 4)
    assert out[0].tolist() == x[:4].mean(dim=0).tolist()
    assert out[1].tolist() == x[4:].mean(dim=0).tolist()

    out = global_max_pool(x, batch)
    assert out.size() == (2, 4)
    assert out[0].tolist() == x[:4].max(dim=0)[0].tolist()
    assert out[1].tolist() == x[4:].max(dim=0)[0].tolist()


def test_permuted_global_pool():
    N_1, N_2 = 4, 6
    x = torch.randn(N_1 + N_2, 4)
    batch = torch.cat([torch.zeros(N_1), torch.ones(N_2)]).to(torch.long)
    perm = torch.randperm(N_1 + N_2)

    out_1 = global_add_pool(x, batch)
    out_2 = global_add_pool(x[perm], batch[perm])
    assert torch.allclose(out_1, out_2)

    out_1 = global_mean_pool(x, batch)
    out_2 = global_mean_pool(x[perm], batch[perm])
    assert torch.allclose(out_1, out_2)

    out_1 = global_max_pool(x, batch)
    out_2 = global_max_pool(x[perm], batch[perm])
    assert torch.allclose(out_1, out_2)
