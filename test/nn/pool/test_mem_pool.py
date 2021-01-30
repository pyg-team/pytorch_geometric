import torch
from torch_geometric.nn import MemPool


def test_mem_pool():
    mpool = MemPool(in_feat=1, out_feat=2)
    q = torch.Tensor([[[1], [2], [3], [4]], [[-2], [2], [-2], [2]],
                      [[-3], [3], [-3], [3]], [[4], [4], [4], [4]],
                      [[5], [5], [5], [5]]])
    mask = torch.Tensor([[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 1],
                         [1, 1, 1, 1], [1, 1, 1, 0]])

    v, C = mpool(q, mask)
    assert v.shape == torch.Size([5, 2, 2])

    loss, P = mpool.computeKl(C, mask)
    assert P.shape == C.shape
