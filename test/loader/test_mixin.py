import subprocess
from time import sleep
import pytest
import torch

from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.testing import (
    onlyLinux,
    onlyNeighborSampler,
)


@onlyLinux
@onlyNeighborSampler
@pytest.mark.parametrize('loader_cores', [None, [1, 2]])
def test_cpu_affinity_neighbor_loader(loader_cores, spawn_context):
    data = Data(x=torch.randn(1, 1))
    loader = NeighborLoader(
        data, num_neighbors=[-1], batch_size=1, num_workers=2
    )
    out = []
    with loader.enable_cpu_affinity(loader_cores):
        iterator = loader._get_iterator()
        workers = iterator._workers
        for worker in workers:
            sleep(2)  # Gives time for worker to initialize.
            process = subprocess.Popen(
                ['taskset', '-c', '-p', f'{worker.pid}'], stdout=subprocess.PIPE
            )
            stdout = process.communicate()[0].decode('utf-8')
            # returns "pid <pid>'s current affinity list <n>-<m>"
            out.append(stdout.split(':')[1].strip())
        if loader_cores:
            out == ['[1]', '[2]']
        else:
            n, m = out[0].split('-')
            assert int(n) == 0
            assert int(out[1].split('-')[0]) == int(m) + 1
            # test if threads assigned to workers are in two consecutive ranges
            # (0-n), (n+1, m)


def init_fn(worker_id):
    assert torch.get_num_threads() == 2
    print(f"{worker_id} uses {torch.get_num_threads()} threads")


@onlyLinux
@onlyNeighborSampler
def test_multithreading_neighbor_loader(spawn_context):
    data = Data(x=torch.randn(1, 1))
    loader = NeighborLoader(
        data,
        num_neighbors=[-1],
        batch_size=1,
        num_workers=2,
        worker_init_fn=init_fn,
    )

    with loader.enable_multithreading(2):
        loader._get_iterator()
        # runs assertion in init_fn
