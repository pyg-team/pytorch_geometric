import subprocess
from time import sleep

import psutil
import pytest
import torch

from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.testing import onlyLinux, onlyNeighborSampler


@pytest.mark.xfail(reason="TODO: Fix test")
@onlyLinux
@onlyNeighborSampler
@pytest.mark.skipif(
    psutil.cpu_count(logical=False) == 1, reason="Requires multiple CPU cores")
@pytest.mark.parametrize('loader_cores', [None, [1, 2]])
def test_cpu_affinity_neighbor_loader(loader_cores, spawn_context):
    data = Data(x=torch.randn(1, 1))
    loader = NeighborLoader(data, num_neighbors=[-1], batch_size=1,
                            num_workers=2)
    out = []
    with loader.enable_cpu_affinity(loader_cores):
        iterator = loader._get_iterator()
        workers = iterator._workers
        sleep(3)  # Gives time for worker to initialize.
        for worker in workers:
            process = subprocess.Popen(
                ['taskset', '-c', '-p', f'{worker.pid}'],
                stdout=subprocess.PIPE)
            stdout = process.communicate()[0].decode('utf-8')
            # returns "pid <pid>'s current affinity list <n>-<m>"
            out.append(stdout.split(':')[1].strip())
        if loader_cores:
            assert out == ['[1]', '[2]']
        else:
            assert out[0] != out[1]


def init_fn(worker_id):
    assert torch.get_num_threads() == 2


@onlyLinux
@onlyNeighborSampler
@pytest.mark.skipif(
    psutil.cpu_count(logical=False) == 1, reason="Requires multiple CPU cores")
def test_multithreading_neighbor_loader(spawn_context):
    loader = NeighborLoader(
        data=Data(x=torch.randn(1, 1)),
        num_neighbors=[-1],
        batch_size=1,
        num_workers=2,
        worker_init_fn=init_fn,
    )

    with loader.enable_multithreading(2):
        loader._get_iterator()  # Runs assertion in `init_fn`.
