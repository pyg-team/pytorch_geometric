import pytest
import torch

import torch_geometric.typing
from torch_geometric.profile import benchmark
from torch_geometric.utils import log_softmax

CALCULATION_VIA_PTR_AVAILABLE = (torch_geometric.typing.WITH_LOG_SOFTMAX
                                 or torch_geometric.typing.WITH_TORCH_SCATTER)

ATOL = 1e-4
RTOL = 1e-4


def test_log_softmax():
    src = torch.tensor([1.0, 1.0, 1.0, 1.0])
    index = torch.tensor([0, 0, 1, 2])
    ptr = torch.tensor([0, 2, 3, 4])

    out = log_softmax(src, index)
    assert torch.allclose(out, torch.tensor([-0.6931, -0.6931, 0.0000,
                                             0.0000]), atol=ATOL, rtol=RTOL)
    if CALCULATION_VIA_PTR_AVAILABLE:
        assert torch.allclose(log_softmax(src, ptr=ptr), out, atol=ATOL,
                              rtol=RTOL)
    else:
        with pytest.raises(NotImplementedError, match="requires 'index'"):
            log_softmax(src, ptr=ptr)

    src = src.view(-1, 1)
    out = log_softmax(src, index)
    assert torch.allclose(
        out,
        torch.tensor([[-0.6931], [-0.6931], [0.0000], [0.0000]]),
        atol=ATOL,
        rtol=RTOL,
    )
    if CALCULATION_VIA_PTR_AVAILABLE:
        assert torch.allclose(log_softmax(src, None, ptr), out, atol=ATOL,
                              rtol=RTOL)

    jit = torch.jit.script(log_softmax)
    assert torch.allclose(jit(src, index), out, atol=ATOL, rtol=RTOL)


def test_log_softmax_backward():
    src_sparse = torch.rand(4, 8, requires_grad=True)
    index = torch.tensor([0, 0, 1, 1])
    src_dense = src_sparse.clone().detach().view(2, 2, src_sparse.size(-1))
    src_dense.requires_grad_(True)

    out_sparse = log_softmax(src_sparse, index)
    out_sparse.sum().backward()
    out_dense = torch.log_softmax(src_dense, dim=1)
    out_dense.sum().backward()

    assert torch.allclose(out_sparse, out_dense.view_as(out_sparse), atol=ATOL)
    assert torch.allclose(src_sparse.grad, src_dense.grad.view_as(src_sparse),
                          atol=ATOL)


def test_log_softmax_dim():
    index = torch.tensor([0, 0, 0, 0])
    ptr = torch.tensor([0, 4])

    src = torch.randn(4)
    assert torch.allclose(
        log_softmax(src, index, dim=0),
        torch.log_softmax(src, dim=0),
        atol=ATOL,
        rtol=RTOL,
    )
    if CALCULATION_VIA_PTR_AVAILABLE:
        assert torch.allclose(
            log_softmax(src, ptr=ptr, dim=0),
            torch.log_softmax(src, dim=0),
            atol=ATOL,
            rtol=RTOL,
        )

    src = torch.randn(4, 16)
    assert torch.allclose(
        log_softmax(src, index, dim=0),
        torch.log_softmax(src, dim=0),
        atol=ATOL,
        rtol=RTOL,
    )
    if CALCULATION_VIA_PTR_AVAILABLE:
        assert torch.allclose(
            log_softmax(src, ptr=ptr, dim=0),
            torch.log_softmax(src, dim=0),
            atol=ATOL,
            rtol=RTOL,
        )

    src = torch.randn(4, 4)
    assert torch.allclose(
        log_softmax(src, index, dim=-1),
        torch.log_softmax(src, dim=-1),
        atol=ATOL,
        rtol=RTOL,
    )
    if CALCULATION_VIA_PTR_AVAILABLE:
        assert torch.allclose(
            log_softmax(src, ptr=ptr, dim=-1),
            torch.log_softmax(src, dim=-1),
            atol=ATOL,
            rtol=RTOL,
        )

    src = torch.randn(4, 4, 16)
    assert torch.allclose(
        log_softmax(src, index, dim=1),
        torch.log_softmax(src, dim=1),
        atol=ATOL,
        rtol=RTOL,
    )
    if CALCULATION_VIA_PTR_AVAILABLE:
        assert torch.allclose(
            log_softmax(src, ptr=ptr, dim=1),
            torch.log_softmax(src, dim=1),
            atol=ATOL,
            rtol=RTOL,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--backward", action="store_true")
    args = parser.parse_args()

    num_nodes, num_edges = 10_000, 200_000
    x = torch.randn(num_edges, 64, device=args.device)
    index = torch.randint(num_nodes, (num_edges, ), device=args.device)

    compiled_log_softmax = torch.compile(log_softmax)

    def dense_softmax(x, index):
        x = x.view(num_nodes, -1, x.size(-1))
        return x.softmax(dim=-1)

    def dense_log_softmax(x, index):
        x = x.view(num_nodes, -1, x.size(-1))
        return torch.log_softmax(x, dim=-1)

    benchmark(
        funcs=[dense_log_softmax, log_softmax, compiled_log_softmax],
        func_names=["Dense Log Softmax", "Vanilla", "Compiled"],
        args=(x, index),
        num_steps=50 if args.device == "cpu" else 500,
        num_warmups=10 if args.device == "cpu" else 100,
        backward=args.backward,
    )
