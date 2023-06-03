import os.path as osp

import torch

from torch_geometric.datasets import FakeDataset, FakeHeteroDataset
from torch_geometric.distributed.partition.partitioner import Partitioner
from torch_geometric.typing import EdgeTypeStr


def test_partition_homo_graph():
    num_partitions = 2
    save_dir = osp.join('./homo', 'partition')
    dataset = FakeDataset()
    data = dataset[0]

    partitioner = Partitioner(output_dir=save_dir, num_parts=num_partitions,
                              data=data)
    partitioner.generate_partition()

    node_map_path = osp.join(save_dir, 'node_map.pt')
    assert osp.exists(node_map_path)
    node_map = torch.load(node_map_path)
    assert node_map.size(0) == data.num_nodes

    edge_map_path = osp.join(save_dir, 'edge_map.pt')
    assert osp.exists(edge_map_path)
    edge_map = torch.load(edge_map_path)
    assert edge_map.size(0) == data.num_edges

    meta_path = osp.join(save_dir, 'META.json')
    assert osp.exists(meta_path)

    graph_store_path1 = osp.join(save_dir, 'part_0', 'graph.pt')
    graph_store_path2 = osp.join(save_dir, 'part_1', 'graph.pt')
    assert osp.exists(graph_store_path1)
    assert osp.exists(graph_store_path2)
    node_feat_path1 = osp.join(save_dir, 'part_0', 'node_feats.pt')
    node_feat_path2 = osp.join(save_dir, 'part_1', 'node_feats.pt')
    assert osp.exists(node_feat_path1)
    assert osp.exists(node_feat_path2)
    graph_store1 = torch.load(graph_store_path1)
    graph_store2 = torch.load(graph_store_path2)
    attr1 = graph_store1.get_all_edge_attrs()
    graph1 = graph_store1.get_edge_index(attr1[0])
    attr2 = graph_store2.get_all_edge_attrs()
    graph2 = graph_store2.get_edge_index(attr2[0])
    assert graph1[0].size(0) + graph2[0].size(0) == data.num_edges
    feature_store1 = torch.load(node_feat_path1)
    node_attrs1 = feature_store1.get_all_tensor_attrs()
    node_feat1 = feature_store1.get_tensor(node_attrs1[0])
    node_id1 = feature_store1.get_global_id(node_attrs1[0])
    feature_store2 = torch.load(node_feat_path2)
    node_attrs2 = feature_store2.get_all_tensor_attrs()
    node_feat2 = feature_store2.get_tensor(node_attrs2[0])
    node_id2 = feature_store2.get_global_id(node_attrs2[0])
    assert node_feat1.size(0) + node_feat2.size(0) == data.num_nodes
    assert torch.allclose(data.x[node_id1], node_feat1)
    assert torch.allclose(data.x[node_id2], node_feat2)


def test_partition_hetero_graph():
    num_partitions = 2
    save_dir = osp.join('./hetero', 'partition')
    dataset = FakeHeteroDataset()
    data = dataset[0]

    partitioner = Partitioner(output_dir=save_dir, num_parts=num_partitions,
                              data=data)
    partitioner.generate_partition()

    meta_path = osp.join(save_dir, 'META.json')
    assert osp.exists(meta_path)

    for k, v in data.num_edges_dict.items():
        assert len(k) == 3
        edge_name = EdgeTypeStr(k)
        edge_map_path = osp.join(save_dir, 'edge_map', f'{edge_name}.pt')
        assert osp.exists(edge_map_path)
        type_edge_map = torch.load(edge_map_path)
        assert type_edge_map.size(0) == v

    for k, v in data.num_nodes_dict.items():
        node_map_path = osp.join(save_dir, 'node_map', f'{k}.pt')
        assert osp.exists(node_map_path)
        type_node_map = torch.load(node_map_path)
        assert type_node_map.size(0) == v

    for pid in range(num_partitions):
        graph_store_path = osp.join(save_dir, f'part_{pid}', 'graph.pt')
        assert osp.exists(graph_store_path)
        node_feat_path = osp.join(save_dir, f'part_{pid}', 'node_feats.pt')
        assert osp.exists(node_feat_path)
