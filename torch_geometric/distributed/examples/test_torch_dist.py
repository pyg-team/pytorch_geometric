import os
import torch
import argparse
import torch.distributed as dist


def main():
    rank = 1
    dist.init_process_group(backend='gloo')

    test_tensor = torch.tensor(rank + 1)

    x = dist.all_reduce(test_tensor, op=dist.ReduceOp.SUM)
    print(x)
    print(f"Test value: {test_tensor.item()}, expected: {sum(range(3))}")


if __name__ == "__main__":
    main()
