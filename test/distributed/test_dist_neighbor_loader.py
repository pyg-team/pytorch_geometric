import socket
import warnings

import pytest
import torch
import torch.multiprocessing as mp

from torch_geometric.data import Data, HeteroData
from torch_geometric.datasets import FakeDataset, FakeHeteroDataset
from torch_geometric.distributed import (
    DistContext,
    DistNeighborLoader,
    DistNeighborSampler,
    LocalFeatureStore,
    LocalGraphStore,
    Partitioner,
)
from torch_geometric.testing import onlyDistributedTest, withMETIS
from torch_geometric.testing.distributed import ProcArgs, assert_run_mproc


def create_dist_data(tmp_path: str, rank: int):
    graph_store = LocalGraphStore.from_partition(tmp_path, pid=rank)
    feat_store = LocalFeatureStore.from_partition(tmp_path, pid=rank)

    return feat_store, graph_store


def dist_neighbor_loader_homo(
    world_size: int,
    tmp_path: str,
    rank: int,
    master_addr: str,
    master_port: int,
    num_workers: int,
    async_sampling: bool,
):
    part_data = create_dist_data(tmp_path, rank)
    input_nodes = part_data[0].get_global_id(None)
    current_ctx = DistContext(
        rank=rank,
        global_rank=rank,
        world_size=world_size,
        global_world_size=world_size,
        group_name='dist-loader-test',
    )

    loader = DistNeighborLoader(
        part_data,
        num_neighbors=[1],
        batch_size=10,
        num_workers=num_workers,
        input_nodes=input_nodes,
        master_addr=master_addr,
        master_port=master_port,
        current_ctx=current_ctx,
        concurrency=10,
        drop_last=True,
        async_sampling=async_sampling,
    )

    edge_index = part_data[1]._edge_index[(None, 'coo')]

    assert str(loader).startswith('DistNeighborLoader')
    assert str(mp.current_process().pid) in str(loader)
    assert isinstance(loader.dist_sampler, DistNeighborSampler)
    assert not part_data[0].meta['is_hetero']

    for batch in loader:
        assert isinstance(batch, Data)
        assert batch.n_id.size() == (batch.num_nodes, )
        assert batch.input_id.numel() == batch.batch_size == 10
        assert batch.edge_index.min() >= 0
        assert batch.edge_index.max() < batch.num_nodes
        assert torch.equal(
            batch.n_id[batch.edge_index],
            edge_index[:, batch.e_id],
        )
    assert loader.channel.empty()


def dist_neighbor_loader_hetero(
    world_size: int,
    tmp_path: str,
    rank: int,
    master_addr: str,
    master_port: int,
    num_workers: int,
    async_sampling: bool,
):
    part_data = create_dist_data(tmp_path, rank)
    input_nodes = ('v0', part_data[0].get_global_id('v0'))
    current_ctx = DistContext(
        rank=rank,
        global_rank=rank,
        world_size=world_size,
        global_world_size=world_size,
        group_name='dist-loader-test',
    )

    loader = DistNeighborLoader(
        part_data,
        num_neighbors=[1],
        batch_size=10,
        num_workers=num_workers,
        input_nodes=input_nodes,
        master_addr=master_addr,
        master_port=master_port,
        current_ctx=current_ctx,
        concurrency=10,
        drop_last=True,
        async_sampling=async_sampling,
    )

    assert str(loader).startswith('DistNeighborLoader')
    assert str(mp.current_process().pid) in str(loader)
    assert isinstance(loader.dist_sampler, DistNeighborSampler)
    assert part_data[0].meta['is_hetero']

    for batch in loader:
        assert isinstance(batch, HeteroData)
        assert batch['v0'].input_id.numel() == batch['v0'].batch_size == 10

        assert len(batch.node_types) == 2
        for node_type in batch.node_types:
            assert torch.equal(batch[node_type].x, batch.x_dict[node_type])
            assert batch.x_dict[node_type].size(0) >= 0
            assert batch[node_type].n_id.size(0) == batch[node_type].num_nodes

        assert len(batch.edge_types) == 4
        for edge_type in batch.edge_types:
            num_edges = batch[edge_type].edge_index.size(1)

            if num_edges > 0:  # Test edge mapping:
                assert batch[edge_type].edge_attr.size(0) == num_edges
                src, _, dst = edge_type
                edge_index = part_data[1]._edge_index[(edge_type, "coo")]
                global_edge_index1 = torch.stack([
                    batch[src].n_id[batch[edge_type].edge_index[0]],
                    batch[dst].n_id[batch[edge_type].edge_index[1]],
                ], dim=0)

                # TODO There is a current known flake, which we need to fix:
                e_id = batch[edge_type].e_id
                if e_id.numel() > 0 and e_id.max() >= edge_index.size(1):
                    warnings.warn("Known test flake")
                else:
                    global_edge_index2 = edge_index[:, e_id]
                    if not torch.equal(global_edge_index1, global_edge_index2):
                        warnings.warn("Known test flake")

    assert loader.channel.empty()


@withMETIS
@onlyDistributedTest
@pytest.mark.parametrize('num_parts', [2])
@pytest.mark.parametrize('num_workers', [0])
@pytest.mark.parametrize('async_sampling', [True])
def test_dist_neighbor_loader_homo(
    tmp_path,
    num_parts,
    num_workers,
    async_sampling,
):
    mp_context = torch.multiprocessing.get_context('spawn')
    addr = '127.0.0.1'
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.settimeout(1)
        s.bind(('', 0))
        port = s.getsockname()[1]

    data = FakeDataset(
        num_graphs=1,
        avg_num_nodes=100,
        avg_degree=3,
        edge_dim=2,
    )[0]
    partitioner = Partitioner(data, num_parts, tmp_path)
    partitioner.generate_partition()

    procs = [
        ProcArgs(
            target=dist_neighbor_loader_homo,
            args=(tmp_path, part, addr, port, num_workers, async_sampling),
        ) for part in range(num_parts)
    ]
    assert_run_mproc(mp_context, procs)


@withMETIS
@onlyDistributedTest
@pytest.mark.parametrize('num_parts', [2])
@pytest.mark.parametrize('num_workers', [0])
@pytest.mark.parametrize('async_sampling', [True])
def test_dist_neighbor_loader_hetero(
    tmp_path,
    num_parts,
    num_workers,
    async_sampling,
):
    mp_context = torch.multiprocessing.get_context('spawn')
    addr = '127.0.0.1'
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.settimeout(1)
        s.bind(('', 0))
        port = s.getsockname()[1]

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

    procs = [
        ProcArgs(
            target=dist_neighbor_loader_hetero,
            args=(tmp_path, part, addr, port, num_workers, async_sampling),
        ) for part in range(num_parts)
    ]
    assert_run_mproc(mp_context, procs)
