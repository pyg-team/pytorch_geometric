from itertools import product

import pytest
import torch

from torch_geometric.nn import LCMAggregation
from torch_geometric.profile import benchmark


def test_lcm_aggregation_with_project():
    x = torch.randn(6, 16)
    index = torch.tensor([0, 0, 1, 1, 1, 2])

    aggr = LCMAggregation(16, 32)
    assert str(aggr) == 'LCMAggregation(16, 32, project=True)'

    out = aggr(x, index)
    assert out.size() == (3, 32)


def test_lcm_aggregation_without_project():
    x = torch.randn(5, 16)
    index = torch.tensor([0, 1, 1, 2, 2])

    aggr = LCMAggregation(16, 16, project=False)
    assert str(aggr) == 'LCMAggregation(16, 16, project=False)'

    out = aggr(x, index)
    assert out.size() == (3, 16)


def test_lcm_aggregation_error_handling():
    with pytest.raises(ValueError, match="must be projected"):
        LCMAggregation(16, 32, project=False)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--backward', action='store_true')
    args = parser.parse_args()

    channels = 128
    batch_size_list = [2**i for i in range(10, 12)]
    num_nodes_list = [2**i for i in range(15, 18)]

    aggr = LCMAggregation(channels, channels, project=False)
    aggr = aggr.to(args.device)

    funcs = []
    func_names = []
    args_list = []
    for batch_size, num_nodes in product(batch_size_list, num_nodes_list):
        x = torch.randn((num_nodes, channels), device=args.device)
        index = torch.randint(0, batch_size, (num_nodes, ), device=args.device)
        index = index.sort()[0]

        funcs.append(aggr)
        func_names.append(f'B={batch_size}, N={num_nodes}')
        args_list.append((x, index))

    benchmark(
        funcs=funcs,
        func_names=func_names,
        args=args_list,
        num_steps=10 if args.device == 'cpu' else 100,
        num_warmups=5 if args.device == 'cpu' else 50,
        backward=args.backward,
        progress_bar=True,
    )
