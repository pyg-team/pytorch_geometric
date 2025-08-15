from itertools import product

import pytest
import torch

import torch_geometric.typing
from torch_geometric.profile import benchmark
from torch_geometric.testing import withCUDA, withDevice, withPackage
from torch_geometric.utils import group_argsort, group_cat, scatter
from torch_geometric.utils._scatter import scatter_argmax


def test_scatter_validate():
    src = torch.randn(100, 32)
    index = torch.randint(0, 10, (100, ), dtype=torch.long)

    with pytest.raises(ValueError, match="must be one-dimensional"):
        scatter(src, index.view(-1, 1))

    with pytest.raises(ValueError, match="must lay between 0 and 1"):
        scatter(src, index, dim=2)

    with pytest.raises(ValueError, match="invalid `reduce` argument 'std'"):
        scatter(src, index, reduce='std')


@withDevice
@withPackage('torch_scatter')
@pytest.mark.parametrize('reduce', ['sum', 'add', 'mean', 'min', 'max'])
def test_scatter(reduce, device):
    import torch_scatter

    src = torch.randn(100, 16, device=device)
    index = torch.randint(0, 8, (100, ), device=device)

    if device.type == 'mps' and reduce in ['min', 'max']:
        with pytest.raises(NotImplementedError, match="for the MPS device"):
            scatter(src, index, dim=0, reduce=reduce)
        return

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


@withDevice
@pytest.mark.parametrize('reduce', ['sum', 'add', 'mean', 'min', 'max'])
def test_scatter_backward(reduce, device):
    src = torch.randn(8, 100, 16, device=device, requires_grad=True)
    index = torch.randint(0, 8, (100, ), device=device)

    if device.type == 'mps' and reduce in ['min', 'max']:
        with pytest.raises(NotImplementedError, match="for the MPS device"):
            scatter(src, index, dim=1, reduce=reduce)
        return

    out = scatter(src, index, dim=1, reduce=reduce)

    assert src.grad is None
    out.mean().backward()
    assert src.grad is not None


@withDevice
def test_scatter_any(device):
    src = torch.randn(6, 4, device=device)
    index = torch.tensor([0, 0, 1, 1, 2, 2], device=device)

    out = scatter(src, index, dim=0, reduce='any')

    for i in range(3):
        for j in range(4):
            assert float(out[i, j]) in src[2 * i:2 * i + 2, j].tolist()


@withDevice
@pytest.mark.parametrize('num_groups', [4])
@pytest.mark.parametrize('descending', [False, True])
def test_group_argsort(num_groups, descending, device):
    src = torch.randn(20, device=device)
    index = torch.randint(0, num_groups, (20, ), device=device)

    out = group_argsort(src, index, 0, num_groups, descending=descending)

    expected = torch.empty_like(index)
    for i in range(num_groups):
        mask = index == i
        tmp = src[mask].argsort(descending=descending)
        perm = torch.empty_like(tmp)
        perm[tmp] = torch.arange(tmp.numel(), device=device)
        expected[mask] = perm

    assert torch.equal(out, expected)

    empty_tensor = torch.tensor([], device=device)
    out = group_argsort(empty_tensor, empty_tensor)
    assert out.numel() == 0


@withCUDA
def test_scatter_argmax(device):
    src = torch.arange(5, device=device)
    index = torch.tensor([2, 2, 0, 0, 3], device=device)

    old_state = torch_geometric.typing.WITH_TORCH_SCATTER
    torch_geometric.typing.WITH_TORCH_SCATTER = False
    argmax = scatter_argmax(src, index, dim_size=6)
    torch_geometric.typing.WITH_TORCH_SCATTER = old_state
    assert argmax.tolist() == [3, 5, 1, 4, 5, 5]


@withDevice
def test_group_cat(device):
    x1 = torch.randn(4, 4, device=device)
    x2 = torch.randn(2, 4, device=device)
    index1 = torch.tensor([0, 0, 1, 2], device=device)
    index2 = torch.tensor([0, 2], device=device)

    expected = torch.cat([x1[:2], x2[:1], x1[2:4], x2[1:]], dim=0)

    out, index = group_cat(
        [x1, x2],
        [index1, index2],
        dim=0,
        return_index=True,
    )
    assert torch.equal(out, expected)
    assert index.tolist() == [0, 0, 0, 1, 2, 2]


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

    from torch_geometric.typing import WITH_TORCH_SCATTER, torch_scatter

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--backward', action='store_true')
    parser.add_argument('--aggr', type=str, default='all')
    args = parser.parse_args()

    num_nodes_list = [4_000, 8_000, 16_000, 32_000, 64_000]

    if args.aggr == 'all':
        aggrs = ['sum', 'mean', 'min', 'max', 'mul']
    else:
        aggrs = args.aggr.split(',')

    def pytorch_scatter(x, index, dim_size, reduce):
        if reduce == 'min' or reduce == 'max':
            reduce = f'a{aggr}'  # `amin` or `amax`
        elif reduce == 'mul':
            reduce = 'prod'
        out = x.new_zeros(dim_size, x.size(-1))
        include_self = reduce in ['sum', 'mean']
        index = index.view(-1, 1).expand(-1, x.size(-1))
        out.scatter_reduce_(0, index, x, reduce, include_self=include_self)
        return out

    def pytorch_index_add(x, index, dim_size, reduce):
        if reduce != 'sum':
            raise NotImplementedError
        out = x.new_zeros(dim_size, x.size(-1))
        out.index_add_(0, index, x)
        return out

    def own_scatter(x, index, dim_size, reduce):
        return torch_scatter.scatter(x, index, dim=0, dim_size=num_nodes,
                                     reduce=reduce)

    def optimized_scatter(x, index, dim_size, reduce):
        return scatter(x, index, dim=0, dim_size=dim_size, reduce=reduce)

    for aggr, num_nodes in product(aggrs, num_nodes_list):
        num_edges = num_nodes * 50
        print(f'aggr: {aggr}, #nodes: {num_nodes}, #edges: {num_edges}')

        x = torch.randn(num_edges, 64, device=args.device)
        index = torch.randint(num_nodes, (num_edges, ), device=args.device)

        funcs = [pytorch_scatter]
        func_names = ['PyTorch scatter_reduce']

        if aggr == 'sum':
            funcs.append(pytorch_index_add)
            func_names.append('PyTorch index_add')

        if WITH_TORCH_SCATTER:
            funcs.append(own_scatter)
            func_names.append('torch_scatter')

        funcs.append(optimized_scatter)
        func_names.append('Optimized PyG Scatter')

        benchmark(
            funcs=funcs,
            func_names=func_names,
            args=(x, index, num_nodes, aggr),
            num_steps=100 if args.device == 'cpu' else 1000,
            num_warmups=50 if args.device == 'cpu' else 500,
            backward=args.backward,
        )
