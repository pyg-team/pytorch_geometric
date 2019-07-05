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
    perm_x = x[perm]
    perm_batch = batch[perm]

    mask_N_1 = perm_batch == 0
    mask_N_2 = perm_batch == 1

    out = global_add_pool(perm_x, perm_batch)
    assert out.size() == (2, 4)
    assert torch.allclose(out[0], perm_x[mask_N_1].sum(dim=0))
    assert torch.allclose(out[1], perm_x[mask_N_2].sum(dim=0))

    out = global_mean_pool(perm_x, perm_batch)
    assert out.size() == (2, 4)
    assert torch.allclose(out[0], perm_x[mask_N_1].mean(dim=0))
    assert torch.allclose(out[1], perm_x[mask_N_2].mean(dim=0))

    out = global_max_pool(perm_x, perm_batch)
    assert out.size() == (2, 4)
    assert torch.allclose(out[0], perm_x[mask_N_1].max(dim=0)[0])
    assert torch.allclose(out[1], perm_x[mask_N_2].max(dim=0)[0])
