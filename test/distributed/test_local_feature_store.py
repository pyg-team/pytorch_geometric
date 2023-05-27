import pytest
import torch

from torch_geometric.distributed import LocalFeatureStore, TensorAttr


def test_local_feature_store_by_graph_name_and_attr_name():
    r""" this test can verify the feature lookup based on the
    ID mapping bewteen global ID and local ID.
    method 1: to put_tensor/get_tensor by graph name and attr name """

    store = LocalFeatureStore()

    group_name = 'partion_1'
    attr_name = 'node_feats'

    whole_feat_data = torch.Tensor([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3],
                                    [4, 4, 4], [5, 5, 5], [6, 6, 6], [7, 7, 7],
                                    [8, 8, 8]])
    # for partion_1 node ids
    ids_part1 = torch.tensor([1, 2, 3, 5, 8, 4], dtype=torch.int64)
    part1_feat_data = whole_feat_data[ids_part1]

    index = torch.arange(0, part1_feat_data.size(0), 1)
    attr = TensorAttr(group_name, attr_name, index)

    # put the part1's id & mapping into feature store for future lookup
    store.set_global_ids_plus_id2index(ids_part1, group_name, attr_name)

    # put the part1's feature in rows [1, 2, 3, 5, 8, 4] into part 1 feature store with increasing order [0, 1, 2 ... 5]
    store.put_tensor(part1_feat_data, attr)

    # lookup the features by global ids like [3, 8, 4]
    local_ids = torch.tensor([3, 8, 4], dtype=torch.int64)

    # get the id-mapping between ids and feature index
    store_id2idx = store.get_id2index(group_name, attr_name)

    # construct ip mapping between global ids and local index
    max_id = torch.max(ids_part1).item()
    id2idx = torch.zeros(max_id + 1, dtype=torch.int64)
    id2idx[ids_part1] = torch.arange(ids_part1.size(0), dtype=torch.int64)

    assert torch.equal(store_id2idx, id2idx)

    # verify the feature looked up
    assert torch.equal(
        store.get_tensor(group_name, attr_name, index=store_id2idx[local_ids]),
        torch.Tensor([[3, 3, 3], [8, 8, 8], [4, 4, 4]]))


def test_local_feature_store_by_all_tensor_attrs():
    r""" this test can verify the feature lookup based on the
    ID mapping bewteen global ID and local ID.
    method 2: to put_tensor/get_tensor by store.get_all_tensor_attrs() """

    store = LocalFeatureStore()

    group_name = 'partion_1'
    attr_name = 'node_feats'

    whole_feat_data = torch.Tensor([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3],
                                    [4, 4, 4], [5, 5, 5], [6, 6, 6], [7, 7, 7],
                                    [8, 8, 8]])
    # for partion_1 node ids
    ids_part1 = torch.tensor([1, 2, 3, 5, 8, 4], dtype=torch.int64)
    part1_feat_data = whole_feat_data[ids_part1]

    index = torch.arange(0, part1_feat_data.size(0), 1)
    attr = TensorAttr(group_name, attr_name, index)

    store.put_tensor(part1_feat_data, group_name, attr_name, index)
    store.set_global_ids_plus_id2index(ids_part1, group_name, attr_name)

    tensor_attrs = store.get_all_tensor_attrs()
    for item in tensor_attrs:
        node_feat = store.get_tensor(item.fully_specify())
        node_ids = store.get_global_ids(item.group_name, item.attr_name)
        node_id2index = store.get_id2index(item.group_name, item.attr_name)

    local_ids = torch.tensor([3, 8, 4], dtype=torch.int64)
    local_feat_index = node_id2index[local_ids]

    assert torch.equal(node_feat[local_feat_index],
                       torch.Tensor([[3, 3, 3], [8, 8, 8], [4, 4, 4]]))

    assert torch.equal(node_ids, ids_part1)
