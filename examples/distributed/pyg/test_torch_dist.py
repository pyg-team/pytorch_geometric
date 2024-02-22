import os
import torch
import argparse
import intel_extension_for_pytorch as ipex
import oneccl_bindings_for_pytorch  # noqa
import torch.distributed as dist
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

num_nodes = 2
master_addr = "10.211.176.210" #DUT1005
master_port = "11111" # '11111'
os.environ["MASTER_ADDR"] = master_addr
os.environ["MASTER_PORT"] = master_port

node_rank = int(os.environ.get("RANK", -1))
mpi_rank = int(os.environ.get("PMI_RANK", -1))
mpi_world_size = int(os.environ.get("PMI_SIZE", -1))
rank = node_rank * mpi_world_size + mpi_rank
world_size = num_nodes * mpi_world_size
# os.environ["RANK"] = str(rank)
os.environ["WORLD_SIZE"] = str(world_size)

logging.info(
    f"node_rank: {node_rank}, mpi_rank: {mpi_rank} -> rank={rank}"
)
logging.info(f"num_nodes: {num_nodes}, mpi_world_size:{mpi_world_size} -> world_size={world_size}")

dist.init_process_group(
    backend="ccl",
    rank=rank,
    world_size=world_size,
    init_method=f"tcp://{master_addr}:{master_port}",
)
device = torch.device(f'xpu:{mpi_rank}')
logging.info(f"{device}: ddp connected")
torch.xpu.synchronize()

test_tensor = torch.tensor(rank).to(device)
x = dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
logging.info(f"node_rank: {node_rank}, mpi_rank: {mpi_rank} -> RESULT received: {x}, (expected: {sum(range(world_size))})")

dist.destroy_process_group()
