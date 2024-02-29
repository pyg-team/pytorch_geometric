
import os
import socket
from torch.multiprocessing import Process
import intel_extension_for_pytorch as ipex
import oneccl_bindings_for_pytorch  # noqa
import torch
import torch.distributed as dist

def run(rank, mpi_rank, size, hostname):
    
    device = torch.device(f'xpu:{mpi_rank}')

    print(f"I am {rank} of {size} in {hostname} running on {device}")
    
    tensor = torch.tensor(rank).to(device)
    dist.barrier()
    if rank == 0:
        tensor += 1
        # Send the tensor to process 1
        dist.send(tensor=tensor, dst=1)
    else:
        # Receive tensor from process 0
        dist.recv(tensor=tensor, src=0)

if __name__ == "__main__":
    mpi_world_size = int(os.environ.get("PMI_SIZE", -1))
    mpi_rank = int(os.environ.get("PMI_RANK", -1))
    node_rank = int(os.environ.get("RANK", -1))
    num_nodes = 1
    
    world_rank = node_rank * mpi_world_size + mpi_rank
    world_size = num_nodes * mpi_world_size
    
    master_addr = "10.211.176.210" #DUT1005
    master_port = "29500" # '11111'
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    hostname = socket.gethostname()
    dist.init_process_group(
        backend="ccl", 
        rank=world_rank, 
        world_size=world_size,
        init_method=f"tcp://{master_addr}:{master_port}",
    )
    run(world_rank, mpi_rank, world_size, hostname)
