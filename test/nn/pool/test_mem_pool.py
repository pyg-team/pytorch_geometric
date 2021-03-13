import torch
from torch_geometric.nn import MemPool


def test_mem_pool():

    mpool = MemPool(heads=3, num_keys=2, in_channels=2, out_channels=3)
    assert mpool.__repr__() == 'MemPool(2, 3, heads=3, keys=2)'
    x = torch.rand((5, 4, 2))
    mask = torch.Tensor([[1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 1],
                         [1, 1, 1, 1], [1, 1, 1, 0]]).bool()
    x, S = mpool(x, mask)
    loss, P = MemPool.kl_loss(S, mask)
    assert x.shape == torch.Size([5, 2, 3])
    assert (S[~mask] == 0).all()
    assert P.shape == S.shape
    assert (P[~mask] == 0).all()

    mpool = MemPool(heads=2, num_keys=2, in_channels=2, out_channels=4)
    assert mpool.__repr__() == 'MemPool(2, 4, heads=2, keys=2)'
    x = torch.rand((4, 2))
    x, S = mpool(x)
    loss, P = MemPool.kl_loss(S)
    assert x.shape == torch.Size([1, 2, 4])
    assert P.shape == S.shape
