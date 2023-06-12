import torch

from torch_geometric.distributed import LocalDataset
from torch_geometric.testing import FakeHeteroDataset


def test_local_dataset():
    dataset = FakeHeteroDataset()
    data = dataset[0]

    # init edge_index and node_features.
    edge_dict = {}
    feature_dict = {}
    edge_ids_dict = {}
    node_ids_dict = {}
    y_dict = {}
    for etype in data.edge_types:
        edge_dict[etype] = data[etype]['edge_index']
        edge_ids_dict[etype] = torch.tensor([1, 2, 3, 5, 7, 4])

    for ntype in data.node_types:
        feature_dict[ntype] = data[ntype].x.clone(
            memory_format=torch.contiguous_format)
        node_ids_dict[ntype] = torch.tensor([1, 2, 3, 5, 8, 4])
        if hasattr(data[ntype], 'y'):
            y_dict[ntype] = data[ntype].y

    # define local_dataset and init_graph() and init_node_features()
    pyg_dataset = LocalDataset()
    pyg_dataset.init_graph(edge_index=edge_dict, edge_ids=edge_ids_dict)
    pyg_dataset.init_node_features(node_feature_data=feature_dict,
                                   ids=node_ids_dict)

    pyg_dataset.init_node_labels(node_label_data=y_dict)
    # get the graph and node_features from the graphstore/featurestore in
    # LocalDataset
    edge_attrs = pyg_dataset.graph.get_all_edge_attrs()

    edge_index = {}
    edge_ids = {}
    for item in edge_attrs:
        edge_index[item.edge_type] = pyg_dataset.graph.get_edge_index(item)
        edge_ids[item.edge_type] = pyg_dataset.graph.get_edge_id(item)

    # Verify the edge_index and edge_ids
    assert (edge_index == edge_dict)
    assert (edge_ids == edge_ids_dict)

    tensor_attrs = pyg_dataset.node_features.get_all_tensor_attrs()
    node_feat = {}
    node_ids = {}
    for item in tensor_attrs:
        node_feat[item.attr_name] = pyg_dataset.node_features.get_tensor(
            item.group_name, item.attr_name)
        node_ids[item.attr_name] = pyg_dataset.node_features.get_global_id(
            item.group_name, item.attr_name)

    # verfiy the node features and node ids.
    assert (node_feat == feature_dict)
    assert (node_ids == node_ids_dict)

    assert torch.equal(y_dict['paper'], pyg_dataset.get_node_label('paper'))
