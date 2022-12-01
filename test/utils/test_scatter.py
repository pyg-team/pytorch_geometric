import pytest
import torch
import torch_scatter

from torch_geometric.testing import withCUDA
from torch_geometric.utils import scatter


def test_scatter_validate():
    src = torch.randn(100, 32)
    index = torch.randint(0, 10, (100, ), dtype=torch.long)

    with pytest.raises(ValueError, match="must be one-dimensional"):
        scatter(src, index.view(-1, 1))

    with pytest.raises(ValueError, match="must lay between 0 and 1"):
        scatter(src, index, dim=2)

    with pytest.raises(ValueError, match="invalid `reduce` argument 'std'"):
        scatter(src, index, reduce='std')


@withCUDA
@pytest.mark.parametrize('reduce', ['sum', 'add', 'mean', 'min', 'max'])
def test_scatter(reduce, device):
    src = torch.randn(100, 16, device=device)
    index = torch.randint(0, 8, (100, ), device=device)

    out1 = scatter(src, index, dim=0, reduce=reduce)
    out2 = torch_scatter.scatter(src, index, dim=0, reduce=reduce)
    assert out1.device == device
    assert torch.allclose(out1, out2, atol=1e-6)

    jit = torch.jit.script(scatter)
    out3 = jit(src, index, dim=0, reduce=reduce)
    assert torch.allclose(out1, out3, atol=1e-6)

    src = torch.randn(8, 100, 16, device=device)
    out1 = scatter(src, index, dim=1, reduce=reduce)
    out2 = torch_scatter.scatter(src, index, dim=1, reduce=reduce)
    assert out1.device == device
    assert torch.allclose(out1, out2, atol=1e-6)


@withCUDA
@pytest.mark.parametrize('reduce', ['sum', 'add', 'mean', 'min', 'max'])
def test_scatter_backward(reduce, device):
    src = torch.randn(8, 100, 16, device=device, requires_grad=True)
    index = torch.randint(0, 8, (100, ), device=device)

    out = scatter(src, index, dim=1, reduce=reduce)

    assert src.grad is None
    out.mean().backward()
    assert src.grad is not None


if __name__ == '__main__':
    # Insights on GPU:
    # ================
    # * "sum": Prefer `scatter_add_` implementation
    # * "mean": Prefer manual implementation via `scatter_add_` + `count`
    # * "min"/"max":
    #   * Prefer `scatter_reduce_` implementation without gradients
    #   * Prefer `torch_sparse` implementation with gradients
    # * "mul": Prefer `torch_sparse` implementation
    #
    # Insights on CPU:
    # ================
    # * "sum": Prefer `scatter_add_` implementation
    # * "mean": Prefer manual implementation via `scatter_add_` + `count`
    # * "min"/"max": Prefer `scatter_reduce_` implementation
    # * "mul" (probably not worth branching for this):
    #   * Prefer `scatter_reduce_` implementation without gradients
    #   * Prefer `torch_sparse` implementation with gradients

    import argparse
    import time

    import torch_scatter

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--backward', action='store_true')
    args = parser.parse_args()

    num_nodes, num_edges, num_feats = 1000, 50000, 64

    num_warmups, num_steps = 500, 1000
    if args.device == 'cpu':
        num_warmups, num_steps = num_warmups // 10, num_steps // 10

    index = torch.randint(num_nodes - 5, (num_edges, ), device=args.device)
    out_grad = torch.randn(num_nodes, num_feats, device=args.device)

    aggrs = ['sum', 'mean', 'min', 'max', 'mul']
    for aggr in aggrs:
        print(f'Aggregator: {aggr}')
        print('==============================')

        reduce = aggr
        if reduce == 'min' or reduce == 'max':
            reduce = f'a{aggr}'  # `amin` or `max`
        elif reduce == 'mul':
            reduce = 'prod'

        t_forward = t_backward = 0
        for i in range(num_warmups + num_steps):
            x = torch.randn(num_edges, num_feats, device=args.device)
            if args.backward:
                x.requires_grad_(True)

            torch.cuda.synchronize()
            t_start = time.perf_counter()

            out = x.new_zeros((num_nodes, num_feats))
            include_self = reduce in ['sum', 'mean']
            broadcasted_index = index.view(-1, 1).expand(-1, num_feats)
            out.scatter_reduce_(0, broadcasted_index, x, reduce,
                                include_self=include_self)

            torch.cuda.synchronize()
            if i >= num_warmups:
                t_forward += time.perf_counter() - t_start

            if args.backward:
                t_start = time.perf_counter()
                out.backward(out_grad)

                torch.cuda.synchronize()
                if i >= num_warmups:
                    t_backward += time.perf_counter() - t_start

        print(f'PyTorch forward:       {t_forward:.4f}s')
        if args.backward:
            print(f'PyTorch backward:      {t_backward:.4f}s')
        print('==============================')

        t_forward = t_backward = 0
        for i in range(num_warmups + num_steps):
            x = torch.randn(num_edges, num_feats, device=args.device)
            if args.backward:
                x.requires_grad_(True)

            torch.cuda.synchronize()
            t_start = time.perf_counter()

            out = torch_scatter.scatter(x, index, dim=0, dim_size=num_nodes,
                                        reduce=aggr)

            torch.cuda.synchronize()
            if i >= num_warmups:
                t_forward += time.perf_counter() - t_start

            if args.backward:
                t_start = time.perf_counter()
                out.backward(out_grad)

                torch.cuda.synchronize()
                if i >= num_warmups:
                    t_backward += time.perf_counter() - t_start

        print(f'torch_sparse forward:  {t_forward:.4f}s')
        if args.backward:
            print(f'torch_sparse backward: {t_backward:.4f}s')
        print('==============================')

        t_forward = t_backward = 0
        for i in range(num_warmups + num_steps):
            x = torch.randn(num_edges, num_feats, device=args.device)
            if args.backward:
                x.requires_grad_(True)

            torch.cuda.synchronize()
            t_start = time.perf_counter()

            out = scatter(x, index, dim=0, dim_size=num_nodes, reduce=aggr)

            torch.cuda.synchronize()
            if i >= num_warmups:
                t_forward += time.perf_counter() - t_start

            if args.backward:
                t_start = time.perf_counter()
                out.backward(out_grad)

                torch.cuda.synchronize()
                if i >= num_warmups:
                    t_backward += time.perf_counter() - t_start

        print(f'torch_sparse forward:  {t_forward:.4f}s')
        if args.backward:
            print(f'torch_sparse backward: {t_backward:.4f}s')

        print()
