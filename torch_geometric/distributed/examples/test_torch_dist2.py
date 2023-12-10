import os
import torch
import argparse
import torch.distributed as dist

os.environ['MASTER_ADDR'] = '10.91.30.105'
os.environ['MASTER_PORT'] = '11112'
rank = int(os.getenv('RANK'))
dist.init_process_group(
    backend='gloo',
    rank=rank,
    world_size=3,
)
test_tensor = torch.tensor(rank)

x = dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
print(x)
print(f"Test value: {test_tensor.item()}, expected: {sum(range(4))}")
