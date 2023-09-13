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
    x = torch.randn(6, 16)
    index = torch.tensor([0, 0, 1, 1, 1, 2])

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
    args = parser.parse_args()

    NS = [2**i for i in range(15, 18+1)]
    BS = [2**i for i in range(10, 12+1)]

    H = 128
    lcm_aggr = LCMAggregation(H, H, project=False)
    lcm_aggr.to(args.device)

    funcs = []
    func_names = []
    args_list = []
    for N, B in product(NS, BS):
        x = torch.randn((N, H), device=args.device)
        index = torch.randint(0, B, (N, ), device=args.device).sort()[0]

        funcs.append(lcm_aggr.forward)
        func_names.append(f'N={N}, B={B}')
        args_list.append((x, index))

    benchmark(
        funcs=funcs,
        func_names=func_names,
        args=args_list,
        num_steps=50 if args.device == 'cpu' else 500,
        num_warmups=10 if args.device == 'cpu' else 100,
        progress_bar=True,
    )
