import torch

from torch_geometric.nn import (
    global_add_pool, global_max_pool, global_mean_pool
)


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

    px = x[perm]
    pbatch = batch[perm]
    px1 = px[pbatch == 0]
    px2 = px[pbatch == 1]

    out = global_add_pool(px, pbatch)
    assert out.size() == (2, 4)
    assert torch.allclose(out[0], px1.sum(dim=0))
    assert torch.allclose(out[1], px2.sum(dim=0))

    out = global_mean_pool(px, pbatch)
    assert out.size() == (2, 4)
    assert torch.allclose(out[0], px1.mean(dim=0))
    assert torch.allclose(out[1], px2.mean(dim=0))

    out = global_max_pool(px, pbatch)
    assert out.size() == (2, 4)
    assert torch.allclose(out[0], px1.max(dim=0)[0])
    assert torch.allclose(out[1], px2.max(dim=0)[0])
