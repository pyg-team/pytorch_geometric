import os
import torch
import argparse
import torch.distributed as dist

os.environ['MASTER_ADDR'] = '10.211.176.217'
os.environ['MASTER_PORT'] = '11111'
node_rank = int(os.environ.get("RANK", -1))
mpi_rank = int(os.environ.get("PMI_RANK", -1))
mpi_world_size = int(os.environ.get("PMI_SIZE", -1))
rank = (node_rank * mpi_world_size + mpi_rank,)
world_size = 2 * mpi_world_size
print(
    f"node_rank: {node_rank}, mpi_rank: {mpi_rank} -> rank={rank}\nmpi_world_size:{mpi_world_size}"
)
dist.init_process_group(
    backend='ccl',
    rank=node_rank * mpi_world_size + mpi_rank,
    world_size=2 * mpi_world_size,
)
test_tensor = torch.tensor(rank)

x = dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
print(x)
print(f"Test value: {test_tensor.item()}, expected: {sum(range(world_size))}")
