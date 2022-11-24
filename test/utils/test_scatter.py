import pytest
import torch
import torch_scatter

from torch_geometric.testing import withPackage
from torch_geometric.utils import scatter


@pytest.mark.parametrize('reduce', ['sum', 'add', 'mean', 'min', 'max', 'mul'])
def test_scatter(reduce):
    torch.manual_seed(12345)

    src = torch.randn(8, 100, 32)
    index = torch.randint(0, 10, (100, ), dtype=torch.long)

    out1 = scatter(src, index, dim=1, reduce=reduce)
    out2 = torch_scatter.scatter(src, index, dim=1, reduce=reduce)
    assert torch.allclose(out1, out2, atol=1e-6)


@pytest.mark.parametrize('reduce', ['sum', 'add', 'mean', 'min', 'max'])
def test_pytorch_scatter_backward(reduce):
    torch.manual_seed(12345)

    src = torch.randn(8, 100, 32).requires_grad_(True)
    index = torch.randint(0, 10, (100, ), dtype=torch.long)

    out = scatter(src, index, dim=1, reduce=reduce).relu()

    assert src.grad is None
    out.mean().backward()
    assert src.grad is not None


@withPackage('torch>=1.12.0')
@pytest.mark.parametrize('reduce', ['min', 'max'])
def test_pytorch_scatter_inplace_backward(reduce):
    torch.manual_seed(12345)

    src = torch.randn(8, 100, 32).requires_grad_(True)
    index = torch.randint(0, 10, (100, ), dtype=torch.long)

    out = scatter(src, index, dim=1, reduce=reduce).relu_()

    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        out.mean().backward()


@pytest.mark.parametrize('reduce', ['sum', 'add', 'min', 'max', 'mul'])
def test_scatter_with_out(reduce):
    torch.manual_seed(12345)

    src = torch.randn(8, 100, 32)
    index = torch.randint(0, 10, (100, ), dtype=torch.long)
    out = torch.randn(8, 10, 32)

    out1 = scatter(src, index, dim=1, out=out.clone(), reduce=reduce)
    out2 = torch_scatter.scatter(src, index, dim=1, out=out.clone(),
                                 reduce=reduce)
    assert torch.allclose(out1, out2, atol=1e-6)


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
    import warnings

    import torch_scatter

    warnings.filterwarnings('ignore', '.*is in beta and the API may change.*')

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

        print()
