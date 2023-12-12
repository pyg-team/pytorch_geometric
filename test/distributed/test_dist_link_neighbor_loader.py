import socket
from typing import Tuple

import pytest
import torch
import torch.multiprocessing as mp

from torch_geometric.data import Data, HeteroData
from torch_geometric.datasets import FakeDataset, FakeHeteroDataset
from torch_geometric.distributed import (
    DistContext,
    DistLinkNeighborLoader,
    DistNeighborSampler,
    LocalFeatureStore,
    LocalGraphStore,
    Partitioner,
)
from torch_geometric.distributed.partition import load_partition_info
from torch_geometric.testing import onlyLinux, withPackage


def create_dist_data(tmp_path: str, rank: int):
    graph_store = LocalGraphStore.from_partition(tmp_path, pid=rank)
    feat_store = LocalFeatureStore.from_partition(tmp_path, pid=rank)
    (
        meta,
        num_partitions,
        partition_idx,
        node_pb,
        edge_pb,
    ) = load_partition_info(tmp_path, rank)

    graph_store.partition_idx = partition_idx
    graph_store.num_partitions = num_partitions
    graph_store.node_pb = node_pb
    graph_store.edge_pb = edge_pb
    graph_store.meta = meta

    feat_store.partition_idx = partition_idx
    feat_store.num_partitions = num_partitions
    feat_store.node_feat_pb = node_pb
    feat_store.edge_feat_pb = edge_pb
    feat_store.meta = meta

    return feat_store, graph_store


def dist_link_neighbor_loader_homo(
    tmp_path: str,
    world_size: int,
    rank: int,
    master_addr: str,
    master_port: int,
    num_workers: int,
    async_sampling: bool,
    neg_ratio: float,
):
    part_data = create_dist_data(tmp_path, rank)
    current_ctx = DistContext(
        rank=rank,
        global_rank=rank,
        world_size=world_size,
        global_world_size=world_size,
        group_name='dist-loader-test',
    )

    edge_label_index = part_data[1].get_edge_index(None, 'coo')
    edge_label = torch.randint(high=2, size=(edge_label_index.size(1), ))

    loader = DistLinkNeighborLoader(
        data=part_data,
        edge_label_index=(None, edge_label_index),
        edge_label=edge_label if neg_ratio is not None else None,
        num_neighbors=[1],
        batch_size=10,
        num_workers=num_workers,
        master_addr=master_addr,
        master_port=master_port,
        current_ctx=current_ctx,
        rpc_worker_names={},
        concurrency=10,
        drop_last=True,
        async_sampling=async_sampling,
    )

    assert str(loader).startswith('DistLinkNeighborLoader')
    assert str(mp.current_process().pid) in str(loader)
    assert isinstance(loader.neighbor_sampler, DistNeighborSampler)
    assert not part_data[0].meta['is_hetero']

    for batch in loader:
        assert isinstance(batch, Data)
        assert batch.n_id.size() == (batch.num_nodes, )
        assert batch.input_id.numel() == batch.batch_size == 10
        assert batch.edge_index.min() >= 0
        assert batch.edge_index.max() < batch.num_nodes


def dist_link_neighbor_loader_hetero(
    tmp_path: str,
    world_size: int,
    rank: int,
    master_addr: str,
    master_port: int,
    num_workers: int,
    async_sampling: bool,
    neg_ratio: float,
    edge_type: Tuple[str, str, str],
):
    part_data = create_dist_data(tmp_path, rank)
    current_ctx = DistContext(
        rank=rank,
        global_rank=rank,
        world_size=world_size,
        global_world_size=world_size,
        group_name="dist-loader-test",
    )

    edge_label_index = part_data[1].get_edge_index(edge_type, 'coo')
    edge_label = torch.randint(high=2, size=(edge_label_index.size(1), ))

    loader = DistLinkNeighborLoader(
        data=part_data,
        edge_label_index=(edge_type, edge_label_index),
        edge_label=edge_label if neg_ratio is not None else None,
        num_neighbors=[1],
        batch_size=10,
        num_workers=num_workers,
        master_addr=master_addr,
        master_port=master_port,
        current_ctx=current_ctx,
        rpc_worker_names={},
        concurrency=10,
        drop_last=True,
        async_sampling=async_sampling,
    )

    assert str(loader).startswith('DistLinkNeighborLoader')
    assert str(mp.current_process().pid) in str(loader)
    assert isinstance(loader.neighbor_sampler, DistNeighborSampler)
    assert part_data[0].meta['is_hetero']

    for batch in loader:
        assert isinstance(batch, HeteroData)
        assert (batch[edge_type].input_id.numel() ==
                batch[edge_type].batch_size == 10)

        assert len(batch.node_types) == 2
        for node_type in batch.node_types:
            assert torch.equal(batch[node_type].x, batch.x_dict[node_type])
            assert batch.x_dict[node_type].size(0) >= 0
            assert batch[node_type].n_id.size(0) == batch[node_type].num_nodes

        assert len(batch.edge_types) == 4
        for edge_type in batch.edge_types:
            assert (batch[edge_type].edge_attr.size(0) ==
                    batch[edge_type].edge_index.size(1))


@onlyLinux
@withPackage('pyg_lib')
@pytest.mark.parametrize('num_parts', [2])
@pytest.mark.parametrize('num_workers', [0])
@pytest.mark.parametrize('async_sampling', [True])
@pytest.mark.parametrize('neg_ratio', [None])
@pytest.mark.skip(reason="'sample_from_edges' not yet implemented")
def test_dist_link_neighbor_loader_homo(
    tmp_path,
    num_parts,
    num_workers,
    async_sampling,
    neg_ratio,
):
    addr = '127.0.0.1'
    mp_context = torch.multiprocessing.get_context('spawn')
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        sock.bind((addr, 0))
        port = sock.getsockname()[1]

    data = FakeDataset(
        num_graphs=1,
        avg_num_nodes=100,
        avg_degree=3,
        edge_dim=2,
    )[0]
    partitioner = Partitioner(data, num_parts, tmp_path)
    partitioner.generate_partition()

    w0 = mp_context.Process(
        target=dist_link_neighbor_loader_homo,
        args=(tmp_path, num_parts, 0, addr, port, num_workers, async_sampling,
              neg_ratio),
    )

    w1 = mp_context.Process(
        target=dist_link_neighbor_loader_homo,
        args=(tmp_path, num_parts, 1, addr, port, num_workers, async_sampling,
              neg_ratio),
    )

    w0.start()
    w1.start()
    w0.join()
    w1.join()


@onlyLinux
@withPackage('pyg_lib')
@pytest.mark.parametrize('num_parts', [2])
@pytest.mark.parametrize('num_workers', [0])
@pytest.mark.parametrize('async_sampling', [True])
@pytest.mark.parametrize('neg_ratio', [None])
@pytest.mark.parametrize('edge_type', [('v0', 'e0', 'v0')])
@pytest.mark.skip(reason="'sample_from_edges' not yet implemented")
def test_dist_link_neighbor_loader_hetero(
    tmp_path,
    num_parts,
    num_workers,
    async_sampling,
    neg_ratio,
    edge_type,
):
    mp_context = torch.multiprocessing.get_context('spawn')
    addr = '127.0.0.1'
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        sock.bind((addr, 0))
        port = sock.getsockname()[1]

    data = FakeHeteroDataset(
        num_graphs=1,
        avg_num_nodes=100,
        avg_degree=3,
        num_node_types=2,
        num_edge_types=4,
        edge_dim=2,
    )[0]
    partitioner = Partitioner(data, num_parts, tmp_path)
    partitioner.generate_partition()

    w0 = mp_context.Process(
        target=dist_link_neighbor_loader_hetero,
        args=(tmp_path, num_parts, 0, addr, port, num_workers, async_sampling,
              neg_ratio, edge_type),
    )

    w1 = mp_context.Process(
        target=dist_link_neighbor_loader_hetero,
        args=(tmp_path, num_parts, 1, addr, port, num_workers, async_sampling,
              neg_ratio, edge_type),
    )

    w0.start()
    w1.start()
    w0.join()
    w1.join()
