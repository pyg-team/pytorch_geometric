import socket
from typing import Dict, List

import torch

import torch_geometric.distributed.rpc as rpc
from torch_geometric.distributed import LocalFeatureStore
from torch_geometric.distributed.dist_context import DistContext, DistRole
from torch_geometric.distributed.rpc import RPCRouter
from torch_geometric.testing import onlyLinux


def run_rpc_feature_test(
    world_size: int,
    rank: int,
    feature: LocalFeatureStore,
    partition_book: torch.Tensor,
    master_port: int,
):
    # 1) Initialize the context info:
    current_ctx = DistContext(
        rank=rank,
        global_rank=rank,
        world_size=world_size,
        global_world_size=world_size,
        group_name='dist-feature-test',
    )
    rpc_worker_names: Dict[DistRole, List[str]] = {}

    rpc.init_rpc(
        current_ctx=current_ctx,
        rpc_worker_names=rpc_worker_names,
        master_addr='localhost',
        master_port=master_port,
    )

    # 2) Collect all workers:
    partition_to_workers = rpc.rpc_partition_to_workers(
        current_ctx, world_size, rank)

    assert partition_to_workers == [
        ['dist-feature-test-0'],
        ['dist-feature-test-1'],
    ]

    # 3) Find the mapping between worker and partition ID:
    rpc_router = RPCRouter(partition_to_workers)

    assert rpc_router.get_to_worker(partition_idx=0) == 'dist-feature-test-0'
    assert rpc_router.get_to_worker(partition_idx=1) == 'dist-feature-test-1'

    meta = {
        'edge_types': None,
        'is_hetero': False,
        'node_types': None,
        'num_parts': 2
    }

    feature.num_partitions = world_size
    feature.partition_idx = rank
    feature.feature_pb = partition_book
    feature.meta = meta
    feature.local_only = False
    feature.set_rpc_router(rpc_router)

    # Global node IDs:
    global_id0 = torch.arange(128 * 2)
    global_id1 = torch.arange(128 * 2) + 128 * 2

    # Lookup the features from stores including locally and remotely:
    tensor0 = feature.lookup_features(global_id0)
    tensor1 = feature.lookup_features(global_id1)

    # Expected searched results:
    cpu_tensor0 = torch.cat([torch.ones(128, 1024), torch.ones(128, 1024) * 2])
    cpu_tensor1 = torch.cat([torch.zeros(128, 1024), torch.zeros(128, 1024)])

    # Verify..
    assert torch.allclose(cpu_tensor0, tensor0.wait())
    assert torch.allclose(cpu_tensor1, tensor1.wait())

    rpc.shutdown_rpc()


@onlyLinux
def test_dist_feature_lookup():
    cpu_tensor0 = torch.cat([torch.ones(128, 1024), torch.ones(128, 1024) * 2])
    cpu_tensor1 = torch.cat([torch.zeros(128, 1024), torch.zeros(128, 1024)])

    # Global node IDs:
    global_id0 = torch.arange(128 * 2)
    global_id1 = torch.arange(128 * 2) + 128 * 2

    # Set the partition book for two features (partition 0 and 1):
    partition_book = torch.cat([
        torch.zeros(128 * 2, dtype=torch.long),
        torch.ones(128 * 2, dtype=torch.long)
    ])

    # Put the test tensor into the different feature stores with IDs:
    feature0 = LocalFeatureStore()
    feature0.put_global_id(global_id0, group_name=None)
    feature0.put_tensor(cpu_tensor0, group_name=None, attr_name='x')

    feature1 = LocalFeatureStore()
    feature1.put_global_id(global_id1, group_name=None)
    feature1.put_tensor(cpu_tensor1, group_name=None, attr_name='x')

    mp_context = torch.multiprocessing.get_context('spawn')
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
    s.close()

    w0 = mp_context.Process(target=run_rpc_feature_test,
                            args=(2, 0, feature0, partition_book, port))
    w1 = mp_context.Process(target=run_rpc_feature_test,
                            args=(2, 1, feature1, partition_book, port))

    w0.start()
    w1.start()
    w0.join()
    w1.join()
