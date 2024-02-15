import atexit
import socket
from typing import Optional

import pytest
import torch

import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.datasets import FakeHeteroDataset
from torch_geometric.distributed import (
    DistNeighborSampler,
    LocalFeatureStore,
    LocalGraphStore,
    Partitioner,
)
from torch_geometric.distributed.dist_context import DistContext
from torch_geometric.distributed.event_loop import ConcurrentEventLoop
from torch_geometric.distributed.rpc import init_rpc, shutdown_rpc
from torch_geometric.sampler import EdgeSamplerInput, NeighborSampler
from torch_geometric.sampler.neighbor_sampler import edge_sample
from torch_geometric.testing import onlyDistributedTest
from torch_geometric.typing import EdgeType


def create_data(rank, world_size, time_attr: Optional[str] = None):
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
        feature_store.put_tensor(data.time, group_name=None, attr_name='time')

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
    other_graph_store = LocalGraphStore.from_partition(tmp_path, int(not rank))
    feature_store = LocalFeatureStore.from_partition(tmp_path, pid=rank)

    return (feature_store, graph_store), other_graph_store


def dist_link_neighbor_sampler(
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
        num_neighbors=[-1, -1],
        shuffle=False,
        disjoint=disjoint,
    )

    # Close RPC & worker group at exit:
    atexit.register(shutdown_rpc)

    init_rpc(
        current_ctx=current_ctx,
        master_addr='localhost',
        master_port=master_port,
    )
    dist_sampler.init_sampler_instance()
    dist_sampler.register_sampler_rpc()
    dist_sampler.event_loop = ConcurrentEventLoop(2)
    dist_sampler.event_loop.start_loop()

    if rank == 0:  # Seed nodes:
        input_row = torch.tensor([1, 6], dtype=torch.int64)
        input_col = torch.tensor([2, 7], dtype=torch.int64)
    else:
        input_row = torch.tensor([4, 9], dtype=torch.int64)
        input_col = torch.tensor([5, 0], dtype=torch.int64)

    inputs = EdgeSamplerInput(
        input_id=None,
        row=input_row,
        col=input_col,
        input_type=None,
    )

    # evaluate distributed edge sample function
    out_dist = dist_sampler.event_loop.run_task(coro=dist_sampler.edge_sample(
        inputs, dist_sampler.node_sample, data.num_nodes, disjoint))

    sampler = NeighborSampler(data=data, num_neighbors=[-1, -1],
                              disjoint=disjoint)

    # Evaluate edge sample function:
    out = edge_sample(
        inputs,
        sampler._sample,
        data.num_nodes,
        disjoint,
        node_time=None,
        neg_sampling=None,
    )

    # Compare distributed output with single machine output:
    assert torch.equal(out_dist.node, out.node)
    assert torch.equal(out_dist.row, out.row)
    assert torch.equal(out_dist.col, out.col)
    if disjoint:
        assert torch.equal(out_dist.batch, out.batch)
    assert out_dist.num_sampled_nodes == out.num_sampled_nodes
    assert out_dist.num_sampled_edges == out.num_sampled_edges


def dist_link_neighbor_sampler_temporal(
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
        num_neighbors=num_neighbors,
        shuffle=False,
        disjoint=True,
        temporal_strategy=temporal_strategy,
        time_attr=time_attr,
    )

    # Close RPC & worker group at exit:
    atexit.register(shutdown_rpc)

    init_rpc(
        current_ctx=current_ctx,
        master_addr='localhost',
        master_port=master_port,
    )
    dist_sampler.init_sampler_instance()
    dist_sampler.register_sampler_rpc()
    dist_sampler.event_loop = ConcurrentEventLoop(2)
    dist_sampler.event_loop.start_loop()

    if rank == 0:  # Seed nodes:
        input_row = torch.tensor([1, 6], dtype=torch.int64)
        input_col = torch.tensor([2, 7], dtype=torch.int64)
    else:
        input_row = torch.tensor([4, 9], dtype=torch.int64)
        input_col = torch.tensor([5, 0], dtype=torch.int64)

    inputs = EdgeSamplerInput(
        input_id=None,
        row=input_row,
        col=input_col,
        time=seed_time,
    )

    # Evaluate distributed edge sample function
    out_dist = dist_sampler.event_loop.run_task(coro=dist_sampler.edge_sample(
        inputs, dist_sampler.node_sample, data.num_nodes, disjoint=True,
        node_time=seed_time, neg_sampling=None))

    sampler = NeighborSampler(
        data=data,
        num_neighbors=num_neighbors,
        disjoint=True,
        temporal_strategy=temporal_strategy,
        time_attr=time_attr,
    )

    # Evaluate edge sample function
    out = edge_sample(
        inputs,
        sampler._sample,
        data.num_nodes,
        disjoint=True,
        node_time=seed_time,
        neg_sampling=None,
    )

    # Compare distributed output with single machine output
    assert torch.equal(out_dist.node, out.node)
    assert torch.equal(out_dist.row, out.row)
    assert torch.equal(out_dist.col, out.col)
    assert torch.equal(out_dist.batch, out.batch)
    assert out_dist.num_sampled_nodes == out.num_sampled_nodes
    assert out_dist.num_sampled_edges == out.num_sampled_edges


def dist_link_neighbor_sampler_hetero(
    data: FakeHeteroDataset,
    tmp_path: str,
    world_size: int,
    rank: int,
    master_port: int,
    input_type: EdgeType,
    disjoint: bool = False,
):
    dist_data, other_graph_store = create_hetero_data(tmp_path, rank)

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
        num_neighbors=[-1],
        shuffle=False,
        disjoint=disjoint,
    )

    # close RPC & worker group at exit:
    atexit.register(shutdown_rpc)

    init_rpc(
        current_ctx=current_ctx,
        master_addr='localhost',
        master_port=master_port,
    )
    dist_sampler.init_sampler_instance()
    dist_sampler.register_sampler_rpc()
    dist_sampler.event_loop = ConcurrentEventLoop(2)
    dist_sampler.event_loop.start_loop()

    # Create input rows/cols such that pairs belong to different partitions.
    # Edge from the current partition:
    edge_label_index1 = dist_data[1]._edge_index[(input_type, 'coo')]
    row_0 = edge_label_index1[0][0]
    col_0 = edge_label_index1[1][0]
    # Edge from the other partition:
    edge_label_index2 = other_graph_store._edge_index[(input_type, 'coo')]
    row_1 = edge_label_index2[0][0]
    col_1 = edge_label_index2[1][0]

    # Seed edges:
    input_row = torch.tensor([row_0, row_1])
    input_col = torch.tensor([col_0, col_1])

    inputs = EdgeSamplerInput(
        input_id=None,
        row=input_row,
        col=input_col,
        input_type=input_type,
    )

    # Evaluate distributed `node_sample` function:
    out_dist = dist_sampler.event_loop.run_task(coro=dist_sampler.edge_sample(
        inputs, dist_sampler.node_sample, data.num_nodes, disjoint))

    sampler = NeighborSampler(
        data=data,
        num_neighbors=[-1],
        disjoint=disjoint,
    )

    # Evaluate edge sample function:
    out = edge_sample(
        inputs,
        sampler._sample,
        data.num_nodes,
        disjoint,
    )

    # Compare distributed output with single machine output:
    for k in data.node_types:
        assert torch.equal(out_dist.node[k].sort()[0], out.node[k].sort()[0])
        assert out_dist.num_sampled_nodes[k] == out.num_sampled_nodes[k]
        if disjoint:
            assert torch.equal(
                out_dist.batch[k].sort()[0],
                out.batch[k].sort()[0],
            )


def dist_link_neighbor_sampler_temporal_hetero(
    data: FakeHeteroDataset,
    tmp_path: str,
    world_size: int,
    rank: int,
    master_port: int,
    input_type: EdgeType,
    seed_time: torch.tensor = None,
    temporal_strategy: str = 'uniform',
    time_attr: str = 'time',
):
    dist_data, other_graph_store = create_hetero_data(tmp_path, rank)

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
        num_neighbors=[-1],
        shuffle=False,
        disjoint=True,
        temporal_strategy=temporal_strategy,
        time_attr=time_attr,
    )

    # close RPC & worker group at exit:
    atexit.register(shutdown_rpc)

    init_rpc(
        current_ctx=current_ctx,
        master_addr='localhost',
        master_port=master_port,
    )
    dist_sampler.init_sampler_instance()
    dist_sampler.register_sampler_rpc()
    dist_sampler.event_loop = ConcurrentEventLoop(2)
    dist_sampler.event_loop.start_loop()

    # Create input rows/cols such that pairs belong to different partitions.
    # Edge from the current partition:
    edge_label_index1 = dist_data[1]._edge_index[(input_type, 'coo')]
    row_0 = edge_label_index1[0][0]
    col_0 = edge_label_index1[1][0]
    # Edge from the other partition:
    edge_label_index2 = other_graph_store._edge_index[(input_type, 'coo')]
    row_1 = edge_label_index2[0][0]
    col_1 = edge_label_index2[1][0]

    # Seed nodes:
    input_row = torch.tensor([row_0, row_1], dtype=torch.int64)
    input_col = torch.tensor([col_0, col_1], dtype=torch.int64)

    inputs = EdgeSamplerInput(
        input_id=None,
        row=input_row,
        col=input_col,
        time=seed_time,
        input_type=input_type,
    )

    # Evaluate distributed node sample function:
    out_dist = dist_sampler.event_loop.run_task(coro=dist_sampler.edge_sample(
        inputs, dist_sampler.node_sample, data.num_nodes, disjoint=True))

    sampler = NeighborSampler(
        data=data,
        num_neighbors=[-1],
        disjoint=True,
        temporal_strategy=temporal_strategy,
        time_attr=time_attr,
    )

    # Evaluate edge sample function:
    out = edge_sample(
        inputs,
        sampler._sample,
        data.num_nodes,
        disjoint=True,
        node_time=seed_time,
        neg_sampling=None,
    )

    # Compare distributed output with single machine output:
    for k in data.node_types:
        assert torch.equal(out_dist.node[k].sort()[0], out.node[k].sort()[0])
        assert torch.equal(out_dist.batch[k].sort()[0], out.batch[k].sort()[0])
        assert out_dist.num_sampled_nodes[k] == out.num_sampled_nodes[k]


@onlyDistributedTest
@pytest.mark.parametrize('disjoint', [False, True])
def test_dist_link_neighbor_sampler(disjoint):
    mp_context = torch.multiprocessing.get_context('spawn')
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]

    world_size = 2
    w0 = mp_context.Process(
        target=dist_link_neighbor_sampler,
        args=(world_size, 0, port, disjoint),
    )

    w1 = mp_context.Process(
        target=dist_link_neighbor_sampler,
        args=(world_size, 1, port, disjoint),
    )

    w0.start()
    w1.start()
    w0.join()
    w1.join()


@onlyDistributedTest
@pytest.mark.parametrize('seed_time', [None, torch.tensor([3, 6])])
@pytest.mark.parametrize('temporal_strategy', ['uniform', 'last'])
def test_dist_link_neighbor_sampler_temporal(seed_time, temporal_strategy):
    mp_context = torch.multiprocessing.get_context('spawn')
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]

    world_size = 2
    w0 = mp_context.Process(
        target=dist_link_neighbor_sampler_temporal,
        args=(world_size, 0, port, seed_time, temporal_strategy, 'time'),
    )

    w1 = mp_context.Process(
        target=dist_link_neighbor_sampler_temporal,
        args=(world_size, 1, port, seed_time, temporal_strategy, 'time'),
    )

    w0.start()
    w1.start()
    w0.join()
    w1.join()


@onlyDistributedTest
@pytest.mark.parametrize('seed_time', [[1, 1], [3, 7]])
@pytest.mark.parametrize('temporal_strategy', ['uniform', 'last'])
def test_dist_link_neighbor_sampler_edge_level_temporal(
    seed_time,
    temporal_strategy,
):
    seed_time = torch.tensor(seed_time)

    mp_context = torch.multiprocessing.get_context('spawn')
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]

    world_size = 2
    w0 = mp_context.Process(
        target=dist_link_neighbor_sampler_temporal,
        args=(world_size, 0, port, seed_time, temporal_strategy, 'edge_time'),
    )

    w1 = mp_context.Process(
        target=dist_link_neighbor_sampler_temporal,
        args=(world_size, 1, port, seed_time, temporal_strategy, 'edge_time'),
    )

    w0.start()
    w1.start()
    w0.join()
    w1.join()


@onlyDistributedTest
@pytest.mark.parametrize('disjoint', [False, True])
def test_dist_link_neighbor_sampler_hetero(tmp_path, disjoint):
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
    data = T.ToUndirected()(data)

    partitioner = Partitioner(data, world_size, tmp_path)
    partitioner.generate_partition()

    w0 = mp_context.Process(
        target=dist_link_neighbor_sampler_hetero,
        args=(data, tmp_path, world_size, 0, port, ('v0', 'e0', 'v0'),
              disjoint),
    )

    w1 = mp_context.Process(
        target=dist_link_neighbor_sampler_hetero,
        args=(data, tmp_path, world_size, 1, port, ('v1', 'e0', 'v0'),
              disjoint),
    )

    w0.start()
    w1.start()
    w0.join()
    w1.join()


@onlyDistributedTest
@pytest.mark.parametrize('seed_time', [None, [0, 0], [3, 3]])
@pytest.mark.parametrize('temporal_strategy', ['uniform', 'last'])
def test_dist_link_neighbor_sampler_temporal_hetero(
    tmp_path,
    seed_time,
    temporal_strategy,
):
    if seed_time is not None:
        seed_time = torch.tensor(seed_time)

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
    data = T.ToUndirected()(data)

    # Add time information to the data:
    data['v0'].time = torch.ones(data['v0'].num_nodes, dtype=torch.int64)
    data['v1'].time = torch.full((data['v1'].num_nodes, ), 2).long()

    partitioner = Partitioner(data, world_size, tmp_path)
    partitioner.generate_partition()

    w0 = mp_context.Process(
        target=dist_link_neighbor_sampler_temporal_hetero,
        args=(data, tmp_path, world_size, 0, port, ('v0', 'e0', 'v0'),
              seed_time, temporal_strategy, 'time'),
    )

    w1 = mp_context.Process(
        target=dist_link_neighbor_sampler_temporal_hetero,
        args=(data, tmp_path, world_size, 1, port, ('v1', 'e0', 'v0'),
              seed_time, temporal_strategy, 'time'),
    )

    w0.start()
    w1.start()
    w0.join()
    w1.join()


@onlyDistributedTest
@pytest.mark.parametrize('seed_time', [[0, 0], [3, 3]])
@pytest.mark.parametrize('temporal_strategy', ['uniform', 'last'])
def test_dist_link_neighbor_sampler_edge_level_temporal_hetero(
    tmp_path,
    seed_time,
    temporal_strategy,
):
    seed_time = torch.tensor(seed_time)

    mp_context = torch.multiprocessing.get_context('spawn')
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
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
    data = T.ToUndirected()(data)

    # Add time information to the data:
    for i, edge_type in enumerate(data.edge_types):
        data[edge_type].edge_time = torch.full(  #
            (data[edge_type].num_edges, ), i, dtype=torch.int64)

    partitioner = Partitioner(data, world_size, tmp_path)
    partitioner.generate_partition()

    w0 = mp_context.Process(
        target=dist_link_neighbor_sampler_temporal_hetero,
        args=(data, tmp_path, world_size, 0, port, ('v0', 'e0', 'v0'),
              seed_time, temporal_strategy, 'edge_time'),
    )

    w1 = mp_context.Process(
        target=dist_link_neighbor_sampler_temporal_hetero,
        args=(data, tmp_path, world_size, 1, port, ('v0', 'e0', 'v1'),
              seed_time, temporal_strategy, 'edge_time'),
    )

    w0.start()
    w1.start()
    w0.join()
    w1.join()
