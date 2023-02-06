import pytest
import torch

from torch_geometric.profile import benchmark
from torch_geometric.testing import withCUDA, withPackage
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
@withPackage('torch_scatter')
@pytest.mark.parametrize('reduce', ['sum', 'add', 'mean', 'min', 'max'])
def test_scatter(reduce, device):
    import torch_scatter

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

    import torch_scatter

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--backward', action='store_true')
    args = parser.parse_args()

    num_nodes, num_edges = 1_000, 50_000
    x = torch.randn(num_edges, 64, device=args.device)
    index = torch.randint(num_nodes, (num_edges, ), device=args.device)

    def pytorch_scatter(x, index, dim_size, reduce):
        if reduce == 'min' or reduce == 'max':
            reduce = f'a{aggr}'  # `amin` or `amax`
        elif reduce == 'mul':
            reduce = 'prod'
        out = x.new_zeros((dim_size, x.size(-1)))
        include_self = reduce in ['sum', 'mean']
        index = index.view(-1, 1).expand(-1, x.size(-1))
        out.scatter_reduce_(0, index, x, reduce, include_self=include_self)
        return out

    def own_scatter(x, index, dim_size, reduce):
        return torch_scatter.scatter(x, index, dim=0, dim_size=num_nodes,
                                     reduce=reduce)

    def optimized_scatter(x, index, dim_size, reduce):
        return scatter(x, index, dim=0, dim_size=dim_size, reduce=reduce)

    aggrs = ['sum', 'mean', 'min', 'max', 'mul']
    for aggr in aggrs:
        print(f'Aggregator: {aggr}')
        benchmark(
            funcs=[pytorch_scatter, own_scatter, optimized_scatter],
            func_names=['PyTorch', 'torch_scatter', 'Optimized'],
            args=(x, index, num_nodes, aggr),
            num_steps=100 if args.device == 'cpu' else 1000,
            num_warmups=50 if args.device == 'cpu' else 500,
            backward=args.backward,
        )
