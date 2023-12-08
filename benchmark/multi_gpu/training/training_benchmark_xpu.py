import os
from typing import Any, Tuple

import intel_extension_for_pytorch as ipex
import oneccl_bindings_for_pytorch  # noqa
import torch.distributed as dist

from benchmark.multi_gpu.training.common import (
    get_predefined_args,
    run,
    supported_sets,
)
from benchmark.utils import get_dataset


def get_dist_params() -> Tuple[int, int, str]:
    master_addr = "127.0.0.1"
    master_port = "29500"
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    mpi_rank = int(os.environ.get("PMI_RANK", -1))
    mpi_world_size = int(os.environ.get("PMI_SIZE", -1))
    rank = mpi_rank if mpi_world_size > 0 else os.environ.get("RANK", 0)
    world_size = (mpi_world_size if mpi_world_size > 0 else os.environ.get(
        "WORLD_SIZE", 1))

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    init_method = f"tcp://{master_addr}:{master_port}"

    return rank, world_size, init_method


def custom_optimizer(model: Any, optimizer: Any) -> Tuple[Any, Any]:
    return ipex.optimize(model, optimizer=optimizer)


if __name__ == '__main__':
    rank, world_size, init_method = get_dist_params()
    dist.init_process_group(backend="ccl", init_method=init_method,
                            world_size=world_size, rank=rank)

    argparser = get_predefined_args()
    args = argparser.parse_args()
    setattr(args, 'device', 'xpu')

    assert args.dataset in supported_sets.keys(), \
        f"Dataset {args.dataset} isn't supported."
    data, num_classes = get_dataset(args.dataset, args.root)

    run(rank, world_size, args, num_classes, data, custom_optimizer)
