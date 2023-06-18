import os.path as osp

import torch

from torch_geometric.datasets import FakeDataset, FakeHeteroDataset
from torch_geometric.distributed import Partitioner
from torch_geometric.typing import EdgeTypeStr
from torch_geometric.distributed import DistDataset

def test_dist_dataset_for_homo():
    num_partitions = 2
    save_dir = osp.join('./homo', 'partition')
    dataset = FakeDataset()
    data = dataset[0]

    partitioner = Partitioner(root=save_dir, num_parts=num_partitions, data=data)
    partitioner.generate_partition()

    data_pidx=0
    dataset0 = DistDataset()
    dataset0.load(
        root_dir=osp.join('./homo', 'partition'),
        partition_idx=data_pidx,
    )
    data_pidx=1
    dataset1 = DistDataset()
    dataset1.load(
        root_dir=osp.join('./homo', 'partition'),
        partition_idx=data_pidx,
    )

    assert data.num_nodes==dataset0.data.num_nodes+dataset1.data.num_nodes
    
    graph_store1 = dataset0.graph
    graph_store2 = dataset1.graph
    attr1 = graph_store1.get_all_edge_attrs()
    graph1 = graph_store1.get_edge_index(attr1[0])
    attr2 = graph_store2.get_all_edge_attrs()
    graph2 = graph_store2.get_edge_index(attr2[0])
    assert graph1[0].size(0) + graph2[0].size(0) == data.num_edges

    feature_store1 = dataset0.node_features
    node_attrs1 = feature_store1.get_all_tensor_attrs()
    node_feat1 = feature_store1.get_tensor(node_attrs1[0].group_name, node_attrs1[0].attr_name)
    node_id1 = feature_store1.get_global_id(node_attrs1[0].group_name)
    feature_store2 = dataset1.node_features
    node_attrs2 = feature_store2.get_all_tensor_attrs()
    node_feat2 = feature_store2.get_tensor(node_attrs2[0].group_name, node_attrs2[0].attr_name)
    node_id2 = feature_store2.get_global_id(node_attrs2[0].group_name)
    assert node_feat1.size(0) + node_feat2.size(0) == data.num_nodes
    assert torch.allclose(data.x[node_id1], node_feat1)
    assert torch.allclose(data.x[node_id2], node_feat2)

def test_dist_dataset_for_hetero():
    num_partitions = 2
    save_dir = osp.join('./hetero', 'partition')
    dataset = FakeHeteroDataset()
    data = dataset[0]

    node_types, edge_types = data.metadata()

    partitioner = Partitioner(root=save_dir, num_parts=num_partitions, data=data)
    partitioner.generate_partition()

    data_pidx=0
    dataset0 = DistDataset()
    dataset0.load(
        root_dir=osp.join('./hetero', 'partition'),
        partition_idx=data_pidx,
    )
    data_pidx=1
    dataset1 = DistDataset()
    dataset1.load(
        root_dir=osp.join('./hetero', 'partition'),
        partition_idx=data_pidx,
    )

    graph_store1 = dataset0.graph
    graph_store2 = dataset1.graph
    
    attr1 = graph_store1.get_all_edge_attrs()    
    attr2 = graph_store2.get_all_edge_attrs()
    assert len(edge_types)==len(attr1)==len(attr2)

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

    feature_store1 = dataset0.node_features
    assert feature_store1 == None
    
    feature_store2 = dataset1.node_features
    assert feature_store2 == None

