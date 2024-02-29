
import os
import socket
from torch.multiprocessing import Process
import intel_extension_for_pytorch as ipex
import oneccl_bindings_for_pytorch  # noqa
import torch
import torch.distributed as dist

def run(rank, size, hostname):

    print(f"I am {rank} of {size} in {hostname}")
    mpi_rank = int(os.environ.get("PMI_RANK", -1))
    device = torch.device(f'xpu:{mpi_rank}')
    msg = torch.tensor(rank).to(device)
    dist.barrier()
    if rank == 0:
        # Send the tensor to process 1 & receive tensor from process 1
        dist.send(tensor=msg, dst=1)
        dist.recv(tensor=msg, src=1)
    else:
        # Send the tensor to process 0 & receive tensor from process 0
        dist.send(tensor=msg, dst=0)
        dist.recv(tensor=msg, src=0)
    dist.barrier()
    print('Rank ', rank, ' has data ', msg)
    print('END')

if __name__ == "__main__":
    size = int(os.environ.get("PMI_SIZE", -1))
    rank = int(os.environ.get("PMI_RANK", -1))
    master_addr = "10.211.176.210" #DUT1005
    master_port = "29500" # '11111'
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    hostname = socket.gethostname()
    dist.init_process_group(
        backend="ccl", 
        rank=rank, 
        world_size=size,
        init_method=f"tcp://{master_addr}:{master_port}",
    )
    run(rank, size, hostname)
    dist.destroy_process_group()