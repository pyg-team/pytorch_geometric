import argparse
import os
from typing import Union

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from benchmark.multi_gpu.training.common import (
    get_predefined_args,
    run,
    supported_sets,
)
from benchmark.utils import get_dataset
from torch_geometric.data import Data, HeteroData


def run_cuda(rank: int, world_size: int, args: argparse.ArgumentParser,
             num_classes: int, data: Union[Data, HeteroData]):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    run(rank, world_size, args, num_classes, data)


if __name__ == '__main__':
    argparser = get_predefined_args()
    argparser.add_argument('--n-gpus', default=1, type=int)
    args = argparser.parse_args()
    setattr(args, 'device', 'cuda')

    assert args.dataset in supported_sets.keys(), \
        f"Dataset {args.dataset} isn't supported."
    data, num_classes = get_dataset(args.dataset, args.root)

    max_world_size = torch.cuda.device_count()
    chosen_world_size = args.n_gpus
    if chosen_world_size <= max_world_size:
        world_size = chosen_world_size
    else:
        print(f'User selected {chosen_world_size} GPUs '
              f'but only {max_world_size} GPUs are available')
        world_size = max_world_size
    print(f'Let\'s use {world_size} GPUs!')

    mp.spawn(
        run_cuda,
        args=(world_size, args, num_classes, data),
        nprocs=world_size,
        join=True,
    )
