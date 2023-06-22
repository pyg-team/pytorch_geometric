import os.path as osp

import pytest
import torch

from torch_geometric.datasets import FakeDataset, FakeHeteroDataset
from torch_geometric.distributed import Partitioner
from torch_geometric.typing import EdgeTypeStr

try:
    # TODO Using `pyg-lib` metis partitioning leads to some weird bugs in the
    # CI. As such, we require `torch-sparse` for these tests for now.
    rowptr = torch.tensor([0, 1])
    col = torch.tensor([0])
    torch.ops.torch_sparse.partition(rowptr, col, None, 1, True)
    WITH_METIS = True
except (AttributeError, RuntimeError):
    WITH_METIS = False


@pytest.mark.skipif(not WITH_METIS, reason='Not compiled with METIS support')
def test_partition_data(tmp_path):
    data = FakeDataset()[0]
    num_parts = 2

    partitioner = Partitioner(data, num_parts, tmp_path)
    partitioner.generate_partition()

    node_map_path = osp.join(tmp_path, 'node_map.pt')
    assert osp.exists(node_map_path)
    node_map = torch.load(node_map_path)
    assert node_map.numel() == data.num_nodes

    edge_map_path = osp.join(tmp_path, 'edge_map.pt')
    assert osp.exists(edge_map_path)
    edge_map = torch.load(edge_map_path)
    assert edge_map.numel() == data.num_edges

    meta_path = osp.join(tmp_path, 'META.json')
    assert osp.exists(meta_path)

    graph0_path = osp.join(tmp_path, 'part_0', 'graph.pt')
    assert osp.exists(graph0_path)
    graph0 = torch.load(graph0_path)
    assert len({'edge_id', 'row', 'col', 'size'} & set(graph0.keys())) == 4

    graph1_path = osp.join(tmp_path, 'part_1', 'graph.pt')
    assert osp.exists(graph1_path)
    graph1 = torch.load(graph1_path)
    assert len({'edge_id', 'row', 'col', 'size'} & set(graph1.keys())) == 4

    node_feats0_path = osp.join(tmp_path, 'part_0', 'node_feats.pt')
    assert osp.exists(node_feats0_path)
    node_feats0 = torch.load(node_feats0_path)

    node_feats1_path = osp.join(tmp_path, 'part_1', 'node_feats.pt')
    assert osp.exists(node_feats1_path)
    node_feats1 = torch.load(node_feats1_path)

    assert (node_feats0['feats']['x'].size(0) +
            node_feats1['feats']['x'].size(0) == data.num_nodes)
    assert torch.equal(data.x[node_feats0['global_id']],
                       node_feats0['feats']['x'])
    assert torch.equal(data.x[node_feats1['global_id']],
                       node_feats1['feats']['x'])


@pytest.mark.skipif(not WITH_METIS, reason='Not compiled with METIS support')
def test_partition_hetero_data(tmp_path):
    data = FakeHeteroDataset()[0]
    num_parts = 2

    partitioner = Partitioner(data, num_parts, tmp_path)
    partitioner.generate_partition()

    meta_path = osp.join(tmp_path, 'META.json')
    assert osp.exists(meta_path)

    for edge_type, num_edges in data.num_edges_dict.items():
        assert len(edge_type) == 3
        edge_name = EdgeTypeStr(edge_type)
        edge_map_path = osp.join(tmp_path, 'edge_map', f'{edge_name}.pt')
        assert osp.exists(edge_map_path)
        edge_map = torch.load(edge_map_path)
        assert edge_map.numel() == num_edges

    for node_type, num_nodes in data.num_nodes_dict.items():
        node_map_path = osp.join(tmp_path, 'node_map', f'{node_type}.pt')
        assert osp.exists(node_map_path)
        node_map = torch.load(node_map_path)
        assert node_map.numel() == num_nodes

    for pid in range(num_parts):
        graph_path = osp.join(tmp_path, f'part_{pid}', 'graph.pt')
        assert osp.exists(graph_path)
        node_feats_path = osp.join(tmp_path, f'part_{pid}', 'node_feats.pt')
        assert osp.exists(node_feats_path)
        edge_feats_path = osp.join(tmp_path, f'part_{pid}', 'edge_feats.pt')
        assert osp.exists(edge_feats_path)
