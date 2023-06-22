import os.path as osp

import torch

from torch_geometric.datasets import FakeDataset, FakeHeteroDataset
from torch_geometric.distributed import (
    LocalFeatureStore,
    LocalGraphStore,
    Partitioner,
)
from torch_geometric.testing import get_random_edge_index


def test_init_homo_graph_from_partition():

    num_partitions = 2
    save_dir = osp.join('./homo', 'partition')
    dataset = FakeDataset()
    data = dataset[0]

    partitioner = Partitioner(root=save_dir, num_parts=num_partitions,
                              data=data)
    partitioner.generate_partition()

    partition_idx = 0
    (meta, num_partitions, partition_idx, graph_store1, node_pb,
     edge_pb) = LocalGraphStore.from_partition('./homo/partition',
                                               partition_idx)

    partition_idx = 1
    (meta, num_partitions, partition_idx, graph_store2, node_pb,
     edge_pb) = LocalGraphStore.from_partition('./homo/partition',
                                               partition_idx)

    attr1 = graph_store1.get_all_edge_attrs()
    attr1 = graph_store1.get_all_edge_attrs()
    graph1 = graph_store1.get_edge_index(attr1[0])
    attr2 = graph_store2.get_all_edge_attrs()
    graph2 = graph_store2.get_edge_index(attr2[0])
    assert graph1[0].size(0) + graph2[0].size(0) == data.num_edges

    partition_idx = 0
    (meta, num_partitions, partition_idx, node_feature_store1,
     edge_feature_store1, node_pb,
     edge_pb) = LocalFeatureStore.from_partition('./homo/partition',
                                                 partition_idx)

    partition_idx = 1
    (meta, num_partitions, partition_idx, node_feature_store2,
     edge_feature_store2, node_pb,
     edge_pb) = LocalFeatureStore.from_partition('./homo/partition',
                                                 partition_idx)

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


def test_init_hetero_graph_from_partition():

    num_partitions = 2
    save_dir = osp.join('./hetero', 'partition')

    dataset = FakeHeteroDataset()
    data = dataset[0]

    node_types, edge_types = data.metadata()

    partitioner = Partitioner(root=save_dir, num_parts=num_partitions,
                              data=data)
    partitioner.generate_partition()

    partition_idx = 0
    (meta, num_partitions, partition_idx, graph_store1, node_pb,
     edge_pb) = LocalGraphStore.from_partition('./hetero/partition',
                                               partition_idx)

    partition_idx = 1
    (meta, num_partitions, partition_idx, graph_store2, node_pb,
     edge_pb) = LocalGraphStore.from_partition('./hetero/partition',
                                               partition_idx)

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
