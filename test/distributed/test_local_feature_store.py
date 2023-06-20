import torch

from torch_geometric.distributed import LocalFeatureStore


def test_local_feature_store_global_id():
    store = LocalFeatureStore()

    feat = torch.Tensor([
        [0, 0, 0],
        [1, 1, 1],
        [2, 2, 2],
        [3, 3, 3],
        [4, 4, 4],
        [5, 5, 5],
        [6, 6, 6],
        [7, 7, 7],
        [8, 8, 8],
    ])

    paper_global_id = torch.tensor([1, 2, 3, 5, 8, 4])
    paper_feat = feat[paper_global_id]

    store.put_global_id(paper_global_id, group_name='paper')
    store.put_tensor(paper_feat, group_name='paper', attr_name='feat')

    out = store.get_tensor_from_global_id(group_name='paper', attr_name='feat',
                                          index=torch.tensor([3, 8, 4]))
    assert torch.equal(out, feat[torch.tensor([3, 8, 4])])


def test_local_feature_store_utils():
    store = LocalFeatureStore()

    feat = torch.Tensor([
        [0, 0, 0],
        [1, 1, 1],
        [2, 2, 2],
        [3, 3, 3],
        [4, 4, 4],
        [5, 5, 5],
        [6, 6, 6],
        [7, 7, 7],
        [8, 8, 8],
    ])

    paper_global_id = torch.tensor([1, 2, 3, 5, 8, 4])
    paper_feat = feat[paper_global_id]

    store.put_tensor(paper_feat, group_name='paper', attr_name='feat')

    assert len(store.get_all_tensor_attrs()) == 1
    attr = store.get_all_tensor_attrs()[0]
    assert attr.group_name == 'paper'
    assert attr.attr_name == 'feat'
    assert attr.index is None
    assert store.get_tensor_size(attr) == (6, 3)


def test_homogeneous_feature_store():
    node_id = torch.randperm(6)
    x = torch.randn(6, 32)
    y = torch.randint(0, 2, (6, ))
    edge_id = torch.randperm(12)
    edge_attr = torch.randn(12, 16)

    store = LocalFeatureStore.from_data(node_id, x, y, edge_id, edge_attr)

    assert len(store.get_all_tensor_attrs()) == 3
    attrs = store.get_all_tensor_attrs()

    assert attrs[0].group_name is None
    assert attrs[0].attr_name == 'x'
    assert attrs[1].group_name is None
    assert attrs[1].attr_name == 'y'
    assert attrs[2].group_name == (None, None)
    assert attrs[2].attr_name == 'edge_attr'

    assert torch.equal(store.get_global_id(group_name=None), node_id)
    assert torch.equal(store.get_tensor(group_name=None, attr_name='x'), x)
    assert torch.equal(store.get_tensor(group_name=None, attr_name='y'), y)
    assert torch.equal(store.get_global_id(group_name=(None, None)), edge_id)
    assert torch.equal(
        store.get_tensor(group_name=(None, None), attr_name='edge_attr'),
        edge_attr,
    )


def test_heterogeneous_feature_store():
    node_type = 'paper'
    edge_type = ('paper', 'to', 'paper')
    node_id_dict = {node_type: torch.randperm(6)}
    x_dict = {node_type: torch.randn(6, 32)}
    y_dict = {node_type: torch.randint(0, 2, (6, ))}
    edge_id_dict = {edge_type: torch.randperm(12)}
    edge_attr_dict = {edge_type: torch.randn(12, 16)}

    store = LocalFeatureStore.from_hetero_data(
        node_id_dict,
        x_dict,
        y_dict,
        edge_id_dict,
        edge_attr_dict,
    )

    assert len(store.get_all_tensor_attrs()) == 3
    attrs = store.get_all_tensor_attrs()

    assert attrs[0].group_name == node_type
    assert attrs[0].attr_name == 'x'
    assert attrs[1].group_name == node_type
    assert attrs[1].attr_name == 'y'
    assert attrs[2].group_name == edge_type
    assert attrs[2].attr_name == 'edge_attr'

    assert torch.equal(
        store.get_global_id(group_name=node_type),
        node_id_dict[node_type],
    )
    assert torch.equal(
        store.get_tensor(group_name=node_type, attr_name='x'),
        x_dict[node_type],
    )
    assert torch.equal(
        store.get_tensor(group_name=node_type, attr_name='y'),
        y_dict[node_type],
    )
    assert torch.equal(
        store.get_global_id(group_name=edge_type),
        edge_id_dict[edge_type],
    )
    assert torch.equal(
        store.get_tensor(group_name=edge_type, attr_name='edge_attr'),
        edge_attr_dict[edge_type],
    )
