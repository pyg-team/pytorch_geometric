import itertools
import warnings

import pytest
import torch
from torch import Tensor

import torch_geometric.typing
from torch_geometric.profile import benchmark
from torch_geometric.testing import withCUDA, withPackage
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import spmm, to_torch_coo_tensor


@withCUDA
@pytest.mark.parametrize('reduce', ['sum', 'mean'])
def test_spmm_basic(device, reduce):
    src = torch.randn(5, 4, device=device)
    other = torch.randn(4, 8, device=device)

    out1 = (src @ other) / (src.size(1) if reduce == 'mean' else 1)
    out2 = spmm(src.to_sparse_csr(), other, reduce=reduce)
    assert out1.size() == (5, 8)
    assert torch.allclose(out1, out2, atol=1e-6)
    if torch_geometric.typing.WITH_TORCH_SPARSE:
        out3 = spmm(SparseTensor.from_dense(src), other, reduce=reduce)
        assert torch.allclose(out2, out3, atol=1e-6)

    # Test `mean` reduction with isolated nodes:
    src[0] = 0.
    out1 = (src @ other) / (4. if reduce == 'mean' else 1.)
    out2 = spmm(src.to_sparse_csr(), other, reduce=reduce)
    assert out1.size() == (5, 8)
    assert torch.allclose(out1, out2, atol=1e-6)
    if torch_geometric.typing.WITH_TORCH_SPARSE:
        out3 = spmm(SparseTensor.from_dense(src), other, reduce=reduce)
        assert torch.allclose(out2, out3, atol=1e-6)


@withCUDA
@withPackage('torch>=2.0.0')
@pytest.mark.parametrize('reduce', ['min', 'max'])
def test_spmm_reduce(device, reduce):
    src = torch.randn(5, 4, device=device)
    other = torch.randn(4, 8, device=device)

    if src.is_cuda:
        with pytest.raises(NotImplementedError, match="not yet supported"):
            spmm(src.to_sparse_csr(), other, reduce)
    else:
        out1 = spmm(src.to_sparse_csr(), other, reduce)
        assert out1.size() == (5, 8)
        if torch_geometric.typing.WITH_TORCH_SPARSE:
            out2 = spmm(SparseTensor.from_dense(src), other, reduce=reduce)
            assert torch.allclose(out1, out2)


@withCUDA
@withPackage('torch>=2.0.0')
@pytest.mark.parametrize(
    'layout', [torch.sparse_coo, torch.sparse_csr, torch.sparse_csc])
@pytest.mark.parametrize('reduce', ['sum', 'mean', 'min', 'max'])
def test_spmm_layout(device, layout, reduce):
    src = torch.randn(5, 4, device=device)
    if layout == torch.sparse_coo:
        src = src.to_sparse_coo()
    elif layout == torch.sparse_csr:
        src = src.to_sparse_csr()
    else:
        assert layout == torch.sparse_csc
        src = src.to_sparse_csc()
    other = torch.randn(4, 8, device=device)

    if src.is_cuda and reduce in {'min', 'max'}:
        with pytest.raises(NotImplementedError, match="not yet supported"):
            spmm(src, other, reduce=reduce)
    elif layout != torch.sparse_csr:
        with pytest.warns(UserWarning, match="Converting sparse tensor"):
            spmm(src, other, reduce=reduce)
    else:
        spmm(src, other, reduce=reduce)


@pytest.mark.parametrize('reduce', ['sum', 'mean'])
def test_spmm_jit(reduce):
    @torch.jit.script
    def jit_torch_sparse(src: SparseTensor, other: Tensor,
                         reduce: str) -> Tensor:
        return spmm(src, other, reduce=reduce)

    @torch.jit.script
    def jit_torch(src: Tensor, other: Tensor, reduce: str) -> Tensor:
        return spmm(src, other, reduce=reduce)

    src = torch.randn(5, 4)
    other = torch.randn(4, 8)

    out1 = src @ other
    out2 = jit_torch(src.to_sparse_csr(), other, reduce)
    assert out1.size() == (5, 8)
    if reduce == 'sum':
        assert torch.allclose(out1, out2, atol=1e-6)
    if torch_geometric.typing.WITH_TORCH_SPARSE:
        out3 = jit_torch_sparse(SparseTensor.from_dense(src), other, reduce)
        assert torch.allclose(out2, out3, atol=1e-6)


if __name__ == '__main__':
    import argparse

    warnings.filterwarnings('ignore', ".*Sparse CSR tensor support.*")
    warnings.filterwarnings('ignore', ".*Converting sparse tensor to CSR.*")

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--backward', action='store_true')
    args = parser.parse_args()

    num_nodes, num_edges = 10_000, 200_000
    x = torch.randn(num_nodes, 64, device=args.device)
    edge_index = torch.randint(num_nodes, (2, num_edges), device=args.device)

    reductions = ['sum', 'mean']
    if not x.is_cuda:
        reductions.extend(['min', 'max'])
    layouts = [torch.sparse_coo, torch.sparse_csr, torch.sparse_csc]

    for reduce, layout in itertools.product(reductions, layouts):
        print(f'Aggregator: {reduce}, Layout: {layout}')

        adj = to_torch_coo_tensor(edge_index, size=num_nodes)
        adj = adj.to_sparse(layout=layout)

        benchmark(
            funcs=[spmm],
            func_names=['spmm'],
            args=(adj, x, reduce),
            num_steps=50 if args.device == 'cpu' else 500,
            num_warmups=10 if args.device == 'cpu' else 100,
            backward=args.backward,
        )
