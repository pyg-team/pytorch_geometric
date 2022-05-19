import torch

from torch_geometric.data import Data
from torch_geometric.testing import withPackage
from torch_geometric.transforms import RootedEgoNets, RootedRWSubgraph


@withPackage('torch_cluster')
def test_rooted_ego_nets():
    x = torch.randn(3, 1)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 1]])
    data = Data(x=x, edge_index=edge_index)
    data_trans = RootedEgoNets(2)(data)

    assert (data_trans.subgraph_batch == 0).sum() == 3
    assert (data_trans.subgraph_batch == 1).sum() == 2
    assert (data_trans.subgraph_batch == 2).sum() == 2

    assert set(data_trans.subgraph_nodes_mapper[torch.where(
        data_trans.subgraph_batch == 0)].numpy()) == {0, 1, 2}
    assert set(data_trans.subgraph_nodes_mapper[torch.where(
        data_trans.subgraph_batch == 1)].numpy()) == {1, 2}
    assert set(data_trans.subgraph_nodes_mapper[torch.where(
        data_trans.subgraph_batch == 2)].numpy()) == {1, 2}

    assert set(data_trans.subgraph_edges_mapper[torch.where(
        data_trans.subgraph_batch == 0)].numpy()) == {0, 1, 2}
    assert set(data_trans.subgraph_edges_mapper[torch.where(
        data_trans.subgraph_batch == 1)].numpy()) == {1, 2}
    assert set(data_trans.subgraph_edges_mapper[torch.where(
        data_trans.subgraph_batch == 2)].numpy()) == {1, 2}

    torch.testing.assert_allclose(data_trans.rooted_subgraph(0).x, data.x)


def test_rooted_rw_subgraph():
    x = torch.randn(3, 1)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 1]])
    data = Data(x=x, edge_index=edge_index)
    data_trans = RootedRWSubgraph(2)(data)

    assert 0 not in set(data_trans.subgraph_nodes_mapper[torch.where(
        data_trans.subgraph_batch == 1)].numpy())
    assert 0 not in set(data_trans.subgraph_nodes_mapper[torch.where(
        data_trans.subgraph_batch == 2)].numpy())
