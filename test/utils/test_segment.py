from itertools import product

import pytest
import torch

import torch_geometric.typing
from torch_geometric.profile import benchmark
from torch_geometric.testing import withCUDA, withoutExtensions
from torch_geometric.utils import scatter, segment
from torch_geometric.utils.sparse import index2ptr


@withCUDA
@withoutExtensions
@pytest.mark.parametrize('reduce', ['sum', 'mean', 'min', 'max'])
def test_segment(device, without_extensions, reduce):
    src = torch.randn(20, 16, device=device)
    ptr = torch.tensor([0, 0, 5, 10, 15, 20], device=device)

    if without_extensions and not torch_geometric.typing.WITH_PT20:
        with pytest.raises(ImportError, match="requires the 'torch-scatter'"):
            segment(src, ptr, reduce=reduce)
    else:
        out = segment(src, ptr, reduce=reduce)

        expected = getattr(torch, reduce)(src.view(4, 5, -1), dim=1)
        expected = expected[0] if isinstance(expected, tuple) else expected

        assert torch.allclose(out[:1], torch.zeros(1, 16, device=device))
        assert torch.allclose(out[1:], expected)


if __name__ == '__main__':
    # Insights on GPU:
    # ================
    # * "mean": Prefer `torch._segment_reduce` implementation
    # * others: Prefer `torch_scatter` implementation
    #
    # Insights on CPU:
    # ================
    # * "all": Prefer `torch_scatter` implementation (but `scatter(...)`
    #          implementation is far superior due to multi-threading usage.
    import argparse

    from torch_geometric.typing import WITH_TORCH_SCATTER, torch_scatter

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--backward', action='store_true')
    parser.add_argument('--aggr', type=str, default='all')
    args = parser.parse_args()

    num_nodes_list = [4_000, 8_000, 16_000, 32_000, 64_000]

    if args.aggr == 'all':
        aggrs = ['sum', 'mean', 'min', 'max']
    else:
        aggrs = args.aggr.split(',')

    def pytorch_segment(x, ptr, reduce):
        if reduce == 'min' or reduce == 'max':
            reduce = f'a{aggr}'  # `amin` or `amax`
        return torch._segment_reduce(x, reduce, offsets=ptr)

    def own_segment(x, ptr, reduce):
        return torch_scatter.segment_csr(x, ptr, reduce=reduce)

    def optimized_scatter(x, index, reduce, dim_size):
        return scatter(x, index, dim=0, dim_size=dim_size, reduce=reduce)

    def optimized_segment(x, index, reduce):
        return segment(x, ptr, reduce=reduce)

    for aggr, num_nodes in product(aggrs, num_nodes_list):
        num_edges = num_nodes * 50
        print(f'aggr: {aggr}, #nodes: {num_nodes}, #edges: {num_edges}')

        x = torch.randn(num_edges, 64, device=args.device)
        index = torch.randint(num_nodes, (num_edges, ), device=args.device)
        index, _ = index.sort()
        ptr = index2ptr(index, size=num_nodes)

        funcs = [pytorch_segment]
        func_names = ['PyTorch segment_reduce']
        arg_list = [(x, ptr, aggr)]

        if WITH_TORCH_SCATTER:
            funcs.append(own_segment)
            func_names.append('torch_scatter')
            arg_list.append((x, ptr, aggr))

        funcs.append(optimized_scatter)
        func_names.append('Optimized PyG Scatter')
        arg_list.append((x, index, aggr, num_nodes))

        funcs.append(optimized_segment)
        func_names.append('Optimized PyG Segment')
        arg_list.append((x, ptr, aggr))

        benchmark(
            funcs=funcs,
            func_names=func_names,
            args=arg_list,
            num_steps=100 if args.device == 'cpu' else 1000,
            num_warmups=50 if args.device == 'cpu' else 500,
            backward=args.backward,
        )
