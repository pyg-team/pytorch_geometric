import os.path as osp

import torch

from torch_geometric.datasets import FakeDataset, FakeHeteroDataset
from torch_geometric.distributed import (
    LocalFeatureStore,
    LocalGraphStore,
    Partitioner,
)
from torch_geometric.testing import withPackage
from torch_geometric.typing import EdgeTypeStr


@withPackage('pyg_lib')
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


@withPackage('pyg_lib')
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


def test_from_partition_data(tmp_path):
    data = FakeDataset()[0]
    num_parts = 2

    partitioner = Partitioner(data, num_parts, tmp_path)
    partitioner.generate_partition()

    partition_idx = 0
    (meta, num_partitions, partition_idx, graph_store1, node_pb,
     edge_pb) = LocalGraphStore.from_partition(tmp_path, partition_idx)

    partition_idx = 1
    (meta, num_partitions, partition_idx, graph_store2, node_pb,
     edge_pb) = LocalGraphStore.from_partition(tmp_path, partition_idx)

    attr1 = graph_store1.get_all_edge_attrs()
    attr1 = graph_store1.get_all_edge_attrs()
    graph1 = graph_store1.get_edge_index(attr1[0])
    attr2 = graph_store2.get_all_edge_attrs()
    graph2 = graph_store2.get_edge_index(attr2[0])
    assert graph1[0].size(0) + graph2[0].size(0) == data.num_edges

    partition_idx = 0
    (meta, num_partitions, partition_idx, node_feature_store1,
     edge_feature_store1, node_pb,
     edge_pb) = LocalFeatureStore.from_partition(tmp_path, partition_idx)

    partition_idx = 1
    (meta, num_partitions, partition_idx, node_feature_store2,
     edge_feature_store2, node_pb,
     edge_pb) = LocalFeatureStore.from_partition(tmp_path, partition_idx)

    node_attrs1 = node_feature_store1.get_all_tensor_attrs()
    node_feat1 = node_feature_store1.get_tensor(node_attrs1[0].group_name,
                                                node_attrs1[0].attr_name)
    node_id1 = node_feature_store1.get_global_id(node_attrs1[0].group_name)

    node_attrs2 = node_feature_store2.get_all_tensor_attrs()
    node_feat2 = node_feature_store2.get_tensor(node_attrs2[0].group_name,
                                                node_attrs2[0].attr_name)
    node_id2 = node_feature_store2.get_global_id(node_attrs2[0].group_name)

    assert node_feat1.size(0) + node_feat2.size(0) == data.num_nodes
    assert torch.allclose(data.x[node_id1], node_feat1)
    assert torch.allclose(data.x[node_id2], node_feat2)


def test_from_partition_hetero_data(tmp_path):
    data = FakeHeteroDataset()[0]
    num_parts = 2

    node_types, edge_types = data.metadata()

    partitioner = Partitioner(data, num_parts, tmp_path)
    partitioner.generate_partition()

    partition_idx = 0
    (meta, num_partitions, partition_idx, graph_store1, node_pb,
     edge_pb) = LocalGraphStore.from_partition(tmp_path, partition_idx)

    partition_idx = 1
    (meta, num_partitions, partition_idx, graph_store2, node_pb,
     edge_pb) = LocalGraphStore.from_partition(tmp_path, partition_idx)

    attr1 = graph_store1.get_all_edge_attrs()
    attr2 = graph_store2.get_all_edge_attrs()
    assert len(edge_types) == len(attr1) == len(attr2)

    ntypes1 = set()
    for attr in attr1:
        ntypes1.add(attr.edge_type[0])
        ntypes1.add(attr.edge_type[2])
    assert len(node_types) == len(ntypes1)

    ntypes2 = set()
    for attr in attr2:
        ntypes2.add(attr.edge_type[0])
        ntypes2.add(attr.edge_type[2])
    assert len(node_types) == len(ntypes2)
