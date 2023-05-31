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

    kwargs = dict(group_name='part1', attr_name='feat')
    part1_global_id = torch.tensor([1, 2, 3, 5, 8, 4])
    part1_feat = feat[part1_global_id]

    store.put_global_id(part1_global_id, **kwargs)
    store.put_tensor(part1_feat, **kwargs)

    out = store.get_tensor_from_global_id(index=torch.tensor([3, 8, 4]),
                                          **kwargs)
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

    kwargs = dict(group_name='part1', attr_name='feat')
    part1_global_id = torch.tensor([1, 2, 3, 5, 8, 4])
    part1_feat = feat[part1_global_id]

    store.put_tensor(part1_feat, **kwargs)

    assert len(store.get_all_tensor_attrs()) == 1
    attr = store.get_all_tensor_attrs()[0]
    assert attr.group_name == 'part1'
    assert attr.attr_name == 'feat'
    assert attr.index is None
    assert store.get_tensor_size(attr) == (6, 3)
