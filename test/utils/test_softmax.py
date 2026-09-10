import pytest
import torch

import torch_geometric.typing
from torch_geometric.profile import benchmark
from torch_geometric.utils import softmax

CALCULATION_VIA_PTR_AVAILABLE = (torch_geometric.typing.WITH_SOFTMAX
                                 or torch_geometric.typing.WITH_TORCH_SCATTER)


def test_softmax():
    src = torch.tensor([1., 1., 1., 1.])
    index = torch.tensor([0, 0, 1, 2])
    ptr = torch.tensor([0, 2, 3, 4])

    out = softmax(src, index)
    assert out.tolist() == [0.5, 0.5, 1, 1]
    assert softmax(src, ptr=ptr).tolist() == out.tolist()

    src = src.view(-1, 1)
    out = softmax(src, index)
    assert out.tolist() == [[0.5], [0.5], [1], [1]]
    assert softmax(src, ptr=ptr).tolist() == out.tolist()

    jit = torch.jit.script(softmax)
    assert torch.allclose(jit(src, index), out)


@pytest.mark.parametrize('sizes', [(2, 2), (3, 1, 2)])
def test_softmax_backward(sizes):
    generator = torch.Generator().manual_seed(0)
    src_sparse = torch.randn(sum(sizes), 8, generator=generator)
    index = torch.arange(len(sizes)).repeat_interleave(torch.tensor(sizes))
    src_dense = src_sparse.clone()
    weight = torch.randn(src_sparse.size(), generator=generator)

    src_sparse.requires_grad_(True)
    src_dense.requires_grad_(True)

    out_sparse = softmax(src_sparse, index)
    out_dense = torch.cat([x.softmax(dim=0) for x in src_dense.split(sizes)])

    # A mean over softmax outputs is constant and gives zero gradients.
    (out_sparse * weight).sum().backward()
    (out_dense * weight).sum().backward()

    assert torch.allclose(out_sparse, out_dense)
    assert torch.isfinite(src_sparse.grad).all()
    assert torch.isfinite(src_dense.grad).all()
    assert not torch.allclose(src_dense.grad, torch.zeros_like(src_dense.grad))
    assert torch.allclose(src_sparse.grad, src_dense.grad)


def test_softmax_dim():
    index = torch.tensor([0, 0, 0, 0])
    ptr = torch.tensor([0, 4])

    src = torch.randn(4)
    assert torch.allclose(softmax(src, index, dim=0), src.softmax(dim=0))
    assert torch.allclose(softmax(src, ptr=ptr, dim=0), src.softmax(dim=0))

    src = torch.randn(4, 16)
    assert torch.allclose(softmax(src, index, dim=0), src.softmax(dim=0))
    assert torch.allclose(softmax(src, ptr=ptr, dim=0), src.softmax(dim=0))

    src = torch.randn(4, 4)
    assert torch.allclose(softmax(src, index, dim=-1), src.softmax(dim=-1))
    if CALCULATION_VIA_PTR_AVAILABLE:
        assert torch.allclose(softmax(src, ptr=ptr, dim=-1), src.softmax(-1))
    else:
        with pytest.raises(ImportError, match="requires the 'torch-scatter'"):
            softmax(src, ptr=ptr, dim=-1)

    src = torch.randn(4, 4, 16)
    assert torch.allclose(softmax(src, index, dim=1), src.softmax(dim=1))
    if CALCULATION_VIA_PTR_AVAILABLE:
        assert torch.allclose(softmax(src, ptr=ptr, dim=1), src.softmax(dim=1))
    else:
        with pytest.raises(ImportError, match="requires the 'torch-scatter'"):
            softmax(src, ptr=ptr, dim=1)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--backward', action='store_true')
    args = parser.parse_args()

    num_nodes, num_edges = 10_000, 200_000
    x = torch.randn(num_edges, 64, device=args.device)
    index = torch.randint(num_nodes, (num_edges, ), device=args.device)

    compiled_softmax = torch.compile(softmax)

    def dense_softmax(x, index):
        x = x.view(num_nodes, -1, x.size(-1))
        return x.softmax(dim=-1)

    benchmark(
        funcs=[dense_softmax, softmax, compiled_softmax],
        func_names=['Dense Softmax', 'Vanilla', 'Compiled'],
        args=(x, index),
        num_steps=50 if args.device == 'cpu' else 500,
        num_warmups=10 if args.device == 'cpu' else 100,
        backward=args.backward,
    )
