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
os.environ["RANK"] = str(rank)
os.environ["WORLD_SIZE"] = str(world_size)

logging.info(
    f"node_rank: {node_rank}, mpi_rank: {mpi_rank} -> rank={rank}"
)
logging.info(f"num_nodes: {num_nodes}, mpi_world_size:{mpi_world_size} -> world_size={world_size}")

dist.init_process_group(
    backend="ccl",
    # rank=rank,
    # world_size=world_size,
    # init_method=f"tcp://{master_addr}:{master_port}",
)

device = torch.device(f'xpu:{mpi_rank}')
my_rank = dist.get_rank()
my_size = dist.get_world_size()
logging.info(f"{device}: ddp connected my rank = {my_rank}  my size = {my_size})")
dist.barrier()


# test_tensor = torch.tensor(rank).to(device)
# logging.info(test_tensor)
# x = dist.all_reduce(test_tensor)
# dist.barrier()
# logging.info(f"node_rank: {node_rank}, mpi_rank: {mpi_rank} -> RESULT received: {x}, (expected: {sum(range(world_size))})")

x = torch.ones([2, 2])
y = torch.ones([4, 4])
with torch.autograd.profiler.profile(record_shapes=True) as prof:
    for _ in range(10):
        dist.all_reduce(x)
        dist.all_reduce(y)
dist.barrier()
print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cpu_time_total"))

dist.destroy_process_group()
