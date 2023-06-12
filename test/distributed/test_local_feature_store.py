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
