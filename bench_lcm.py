from itertools import product

import pytest
import torch

from torch_geometric.nn import LCMAggregation
from torch_geometric.profile import benchmark

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    # BS = [2**i for i in range(10, 12+1)]
    BS = [2**12]
    NS = [2**i for i in range(10, 16+1)]

    H = 128
    lcm_aggr = LCMAggregation(H, H, project=False)
    lcm_aggr.to(args.device)

    funcs = []
    func_names = []
    args_list = []
    for B, N in product(BS, NS):
        x = torch.randn((N, H), device=args.device)
        index = torch.randint(0, B, (N, ), device=args.device).sort()[0]

        funcs.append(lcm_aggr.forward)
        func_names.append(f'B={B}, N={N}')
        args_list.append((x, index))

    benchmark(
        funcs=funcs,
        func_names=func_names,
        args=args_list,
        num_steps=50 if args.device == 'cpu' else 500,
        num_warmups=10 if args.device == 'cpu' else 100,
        progress_bar=True,
    )