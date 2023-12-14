import atexit
import socket
from typing import Optional

import pytest
import torch

from torch_geometric.data import Data
from torch_geometric.datasets import FakeHeteroDataset
from torch_geometric.distributed import (
    LocalFeatureStore,
    LocalGraphStore,
    Partitioner,
)
from torch_geometric.distributed.dist_context import DistContext
from torch_geometric.distributed.dist_neighbor_sampler import (
    DistNeighborSampler,
    close_sampler,
)
from torch_geometric.distributed.partition import load_partition_info
from torch_geometric.distributed.rpc import init_rpc
from torch_geometric.sampler import NeighborSampler, NodeSamplerInput
from torch_geometric.sampler.neighbor_sampler import node_sample
from torch_geometric.testing import onlyLinux, withPackage


def create_data(rank: int, world_size: int, time_attr: Optional[str] = None):
    if rank == 0:  # Partition 0:
        node_id = torch.tensor([0, 1, 2, 3, 4, 5, 9])
        edge_index = torch.tensor([  # Sorted by destination.
            [1, 2, 3, 4, 5, 0, 0],
            [0, 1, 2, 3, 4, 4, 9],
        ])
    else:  # Partition 1:
        node_id = torch.tensor([0, 4, 5, 6, 7, 8, 9])
        edge_index = torch.tensor([  # Sorted by destination.
            [5, 6, 7, 8, 9, 5, 0],
            [4, 5, 6, 7, 8, 9, 9],
        ])
    feature_store = LocalFeatureStore.from_data(node_id)
    graph_store = LocalGraphStore.from_data(
        edge_id=None,
        edge_index=edge_index,
        num_nodes=10,
        is_sorted=True,
    )

    graph_store.node_pb = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    graph_store.meta.update({'num_parts': 2})
    graph_store.partition_idx = rank
    graph_store.num_partitions = world_size

    edge_index = torch.tensor([  # Create reference data:
        [1, 2, 3, 4, 5, 0, 5, 6, 7, 8, 9, 0],
        [0, 1, 2, 3, 4, 4, 9, 5, 6, 7, 8, 9],
    ])
    data = Data(x=None, y=None, edge_index=edge_index, num_nodes=10)

    if time_attr == 'time':  # Create node-level time data:
        data.time = torch.tensor([5, 0, 1, 3, 3, 4, 4, 4, 4, 4])
        feature_store.put_tensor(data.time, group_name=None,
                                 attr_name=time_attr)

    elif time_attr == 'edge_time':  # Create edge-level time data:
        data.edge_time = torch.tensor([0, 1, 2, 3, 4, 5, 7, 7, 7, 7, 7, 11])

        if rank == 0:
            edge_time = torch.tensor([0, 1, 2, 3, 4, 5, 11])
        if rank == 1:
            edge_time = torch.tensor([4, 7, 7, 7, 7, 7, 11])

        feature_store.put_tensor(edge_time, group_name=None,
                                 attr_name=time_attr)

    return (feature_store, graph_store), data


def create_hetero_data(tmp_path: str, rank: int):
    graph_store = LocalGraphStore.from_partition(tmp_path, pid=rank)
    feature_store = LocalFeatureStore.from_partition(tmp_path, pid=rank)

    out = load_partition_info(tmp_path, rank)

    feature_store.meta = graph_store.meta = out[0]
    feature_store.num_partitions = graph_store.num_partitions = out[1]
    feature_store.partition_idx = graph_store.partition_idx = out[2]
    feature_store.node_feat_pb = graph_store.node_pb = out[3]
    feature_store.edge_feat_pb = graph_store.edge_pb = out[4]

    return feature_store, graph_store


def dist_neighbor_sampler(
    world_size: int,
    rank: int,
    master_port: int,
    disjoint: bool = False,
):
    dist_data, data = create_data(rank, world_size)

    current_ctx = DistContext(
        rank=rank,
        global_rank=rank,
        world_size=world_size,
        global_world_size=world_size,
        group_name='dist-sampler-test',
    )

    dist_sampler = DistNeighborSampler(
        data=dist_data,
        current_ctx=current_ctx,
        rpc_worker_names={},
        num_neighbors=[-1, -1],
        shuffle=False,
        disjoint=disjoint,
    )
    # Close RPC & worker group at exit:
    atexit.register(close_sampler, 0, dist_sampler)

    init_rpc(
        current_ctx=current_ctx,
        rpc_worker_names={},
        master_addr='localhost',
        master_port=master_port,
    )

    dist_sampler.register_sampler_rpc()
    dist_sampler.init_event_loop()

    if rank == 0:  # Seed nodes:
        input_node = torch.tensor([1, 6])
    else:
        input_node = torch.tensor([4, 9])

    inputs = NodeSamplerInput(input_id=None, node=input_node)

    # Evaluate distributed node sample function:
    out_dist = dist_sampler.event_loop.run_task(
        coro=dist_sampler.node_sample(inputs))

    sampler = NeighborSampler(
        data=data,
        num_neighbors=[-1, -1],
        disjoint=disjoint,
    )

    # Evaluate node sample function:
    out = node_sample(inputs, sampler._sample)

    # Compare distributed output with single machine output:
    assert torch.equal(out_dist.node, out.node)
    assert torch.equal(out_dist.row, out.row)
    assert torch.equal(out_dist.col, out.col)
    if disjoint:
        assert torch.equal(out_dist.batch, out.batch)
    assert out_dist.num_sampled_nodes == out.num_sampled_nodes
    assert out_dist.num_sampled_edges == out.num_sampled_edges


def dist_neighbor_sampler_temporal(
    world_size: int,
    rank: int,
    master_port: int,
    seed_time: torch.tensor = None,
    temporal_strategy: str = 'uniform',
    time_attr: str = 'time',
):
    dist_data, data = create_data(rank, world_size, time_attr)

    current_ctx = DistContext(
        rank=rank,
        global_rank=rank,
        world_size=world_size,
        global_world_size=world_size,
        group_name='dist-sampler-test',
    )

    num_neighbors = [-1, -1] if temporal_strategy == 'uniform' else [1, 1]
    dist_sampler = DistNeighborSampler(
        data=dist_data,
        current_ctx=current_ctx,
        rpc_worker_names={},
        num_neighbors=num_neighbors,
        shuffle=False,
        disjoint=True,
        temporal_strategy=temporal_strategy,
        time_attr=time_attr,
    )
    # Close RPC & worker group at exit:
    atexit.register(close_sampler, 0, dist_sampler)

    init_rpc(
        current_ctx=current_ctx,
        rpc_worker_names={},
        master_addr='localhost',
        master_port=master_port,
    )
    dist_sampler.register_sampler_rpc()
    dist_sampler.init_event_loop()

    if rank == 0:  # Seed nodes:
        input_node = torch.tensor([1, 6], dtype=torch.int64)
    else:
        input_node = torch.tensor([4, 9], dtype=torch.int64)

    inputs = NodeSamplerInput(
        input_id=None,
        node=input_node,
        time=seed_time,
    )

    # Evaluate distributed node sample function:
    out_dist = dist_sampler.event_loop.run_task(
        coro=dist_sampler.node_sample(inputs))
    sampler = NeighborSampler(
        data=data,
        num_neighbors=num_neighbors,
        disjoint=True,
        temporal_strategy=temporal_strategy,
        time_attr=time_attr,
    )

    # Evaluate node sample function:
    out = node_sample(inputs, sampler._sample)

    # Compare distributed output with single machine output:
    assert torch.equal(out_dist.node, out.node)
    assert torch.equal(out_dist.row, out.row)
    assert torch.equal(out_dist.col, out.col)
    assert torch.equal(out_dist.batch, out.batch)
    assert out_dist.num_sampled_nodes == out.num_sampled_nodes
    assert out_dist.num_sampled_edges == out.num_sampled_edges


def dist_neighbor_sampler_hetero(
    data: FakeHeteroDataset,
    tmp_path: str,
    world_size: int,
    rank: int,
    master_port: int,
    input_type: str,
    disjoint: bool = False,
):
    dist_data = create_hetero_data(tmp_path, rank)

    current_ctx = DistContext(
        rank=rank,
        global_rank=rank,
        world_size=world_size,
        global_world_size=world_size,
        group_name='dist-sampler-test',
    )

    num_neighbors = [-1, -1]
    dist_sampler = DistNeighborSampler(
        data=dist_data,
        current_ctx=current_ctx,
        rpc_worker_names={},
        num_neighbors=num_neighbors,
        shuffle=False,
        disjoint=disjoint,
    )

    # Close RPC & worker group at exit:
    atexit.register(close_sampler, 0, dist_sampler)

    init_rpc(
        current_ctx=current_ctx,
        rpc_worker_names={},
        master_addr='localhost',
        master_port=master_port,
    )

    dist_sampler.register_sampler_rpc()
    dist_sampler.init_event_loop()

    # Create inputs nodes such that each belongs to a different partition:
    node_pb_list = dist_data[1].node_pb[input_type].tolist()
    node_0 = node_pb_list.index(0)
    node_1 = node_pb_list.index(1)

    input_node = torch.tensor([node_0, node_1], dtype=torch.int64)

    inputs = NodeSamplerInput(
        input_id=None,
        node=input_node,
        input_type=input_type,
    )

    # Evaluate distributed node sample function:
    out_dist = dist_sampler.event_loop.run_task(
        coro=dist_sampler.node_sample(inputs))

    sampler = NeighborSampler(
        data=data,
        num_neighbors=num_neighbors,
        disjoint=disjoint,
    )

    # Evaluate node sample function:
    out = node_sample(inputs, sampler._sample)

    # Compare distributed output with single machine output:
    for k in data.node_types:
        assert torch.equal(out_dist.node[k].sort()[0], out.node[k].sort()[0])
        assert out_dist.num_sampled_nodes[k] == out.num_sampled_nodes[k]
        if disjoint:
            assert torch.equal(
                out_dist.batch[k].sort()[0],
                out.batch[k].sort()[0],
            )


@onlyLinux
@withPackage('pyg_lib')
@pytest.mark.parametrize('disjoint', [False, True])
def test_dist_neighbor_sampler(disjoint):
    mp_context = torch.multiprocessing.get_context('spawn')
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]

    world_size = 2
    w0 = mp_context.Process(
        target=dist_neighbor_sampler,
        args=(world_size, 0, port, disjoint),
    )

    w1 = mp_context.Process(
        target=dist_neighbor_sampler,
        args=(world_size, 1, port, disjoint),
    )

    w0.start()
    w1.start()
    w0.join()
    w1.join()


@onlyLinux
@withPackage('pyg_lib')
@pytest.mark.parametrize('seed_time', [None, torch.tensor([3, 6])])
@pytest.mark.parametrize('temporal_strategy', ['uniform'])
def test_dist_neighbor_sampler_temporal(seed_time, temporal_strategy):
    mp_context = torch.multiprocessing.get_context('spawn')
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]

    world_size = 2
    w0 = mp_context.Process(
        target=dist_neighbor_sampler_temporal,
        args=(world_size, 0, port, seed_time, temporal_strategy, 'time'),
    )

    w1 = mp_context.Process(
        target=dist_neighbor_sampler_temporal,
        args=(world_size, 1, port, seed_time, temporal_strategy, 'time'),
    )

    w0.start()
    w1.start()
    w0.join()
    w1.join()


@onlyLinux
@withPackage('pyg_lib')
@pytest.mark.parametrize('seed_time', [[3, 7]])
@pytest.mark.parametrize('temporal_strategy', ['last'])
def test_dist_neighbor_sampler_edge_level_temporal(
    seed_time,
    temporal_strategy,
):
    seed_time = torch.tensor(seed_time)

    mp_context = torch.multiprocessing.get_context('spawn')
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]

    world_size = 2
    w0 = mp_context.Process(
        target=dist_neighbor_sampler_temporal,
        args=(world_size, 0, port, seed_time, temporal_strategy, 'edge_time'),
    )

    w1 = mp_context.Process(
        target=dist_neighbor_sampler_temporal,
        args=(world_size, 1, port, seed_time, temporal_strategy, 'edge_time'),
    )

    w0.start()
    w1.start()
    w0.join()
    w1.join()


@withPackage('pyg_lib')
@pytest.mark.parametrize('disjoint', [False, True])
def test_dist_neighbor_sampler_hetero(tmp_path, disjoint):
    mp_context = torch.multiprocessing.get_context('spawn')
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]

    world_size = 2
    data = FakeHeteroDataset(
        num_graphs=1,
        avg_num_nodes=100,
        avg_degree=3,
        num_node_types=2,
        num_edge_types=4,
        edge_dim=2,
    )[0]

    partitioner = Partitioner(data, world_size, tmp_path)
    partitioner.generate_partition()

    w0 = mp_context.Process(
        target=dist_neighbor_sampler_hetero,
        args=(data, tmp_path, world_size, 0, port, 'v0', disjoint),
    )

    w1 = mp_context.Process(
        target=dist_neighbor_sampler_hetero,
        args=(data, tmp_path, world_size, 1, port, 'v1', disjoint),
    )

    w0.start()
    w1.start()
    w0.join()
    w1.join()
