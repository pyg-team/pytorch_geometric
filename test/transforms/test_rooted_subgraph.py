import torch

from torch_geometric.data import Data, DataLoader
from torch_geometric.testing import withPackage
from torch_geometric.transforms import RootedEgoNets, RootedRWSubgraph


def test_rooted_ego_nets():

    x = torch.randn(3, 1)
    edge_attr = torch.randn(4, 3)
    edge_index = torch.tensor([[0, 1, 2, 2], [1, 2, 1, 2]])
    data = Data(x=x, edge_attr=edge_attr, edge_index=edge_index)
    sub_data = RootedEgoNets(2)(data)

    assert (sub_data.n_sub_batch == 0).sum() == 3
    assert (sub_data.n_sub_batch == 1).sum() == 2
    assert (sub_data.n_sub_batch == 2).sum() == 2

    assert sub_data.n_id[sub_data.n_sub_batch == 0].tolist() == [0, 1, 2]
    assert sub_data.n_id[sub_data.n_sub_batch == 1].tolist() == [1, 2]
    assert sub_data.n_id[sub_data.n_sub_batch == 2].tolist() == [1, 2]

    assert (sub_data.e_sub_batch == 0).sum() == 4
    assert (sub_data.e_sub_batch == 1).sum() == 3
    assert (sub_data.e_sub_batch == 2).sum() == 3

    assert sub_data.e_id[sub_data.e_sub_batch == 0].tolist() == [0, 1, 2, 3]
    assert sub_data.e_id[sub_data.e_sub_batch == 1].tolist() == [1, 2, 3]
    assert sub_data.e_id[sub_data.e_sub_batch == 2].tolist() == [1, 2, 3]

    feat_data = sub_data.map_data()

    assert torch.allclose(feat_data.x, x[[0, 1, 2, 1, 2, 1, 2], :])
    assert torch.allclose(feat_data.edge_attr,
                          edge_attr[[0, 1, 2, 3, 1, 2, 3, 1, 2, 3], :])
    assert torch.allclose(
        feat_data.edge_index,
        torch.tensor([[0, 1, 2, 2, 3, 4, 4, 5, 6, 6],
                      [1, 2, 1, 2, 4, 3, 4, 6, 5, 6]]))


@withPackage('torch_cluster')
def test_rooted_rw_subgraph():
    x = torch.randn(3, 1)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 1]])
    data = Data(x=x, edge_index=edge_index)
    data_trans = RootedRWSubgraph(2)(data)
    data_trans.map_features()

    assert 0 not in set(data_trans.subgraph_nodes_mapper[torch.where(
        data_trans.subgraph_batch == 1)].numpy())
    assert 0 not in set(data_trans.subgraph_nodes_mapper[torch.where(
        data_trans.subgraph_batch == 2)].numpy())


def test_rooted_subgraph_minibatch():

    x = torch.randn(2, 1)
    edge_index = torch.tensor([[0], [1]])
    data = Data(x=x, edge_index=edge_index)
    sub_data = RootedEgoNets(2)(data)

    loader = DataLoader([sub_data, sub_data], batch_size=2)
    batch = next(iter(loader))

    assert torch.allclose(batch.n_id, torch.tensor([0, 1, 1, 2, 3, 3]))
    assert torch.allclose(batch.e_id, torch.tensor([0, 1]))
    assert torch.allclose(batch.n_sub_batch, torch.tensor([0, 0, 1, 2, 2, 3]))
    assert torch.allclose(batch.e_sub_batch, torch.tensor([0, 2]))
