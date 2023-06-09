import pytest
import torch

import torch_geometric.typing
from torch_geometric.data import Data
from torch_geometric.loader import ClusterData, ClusterLoader
from torch_geometric.testing import onlyFullTest
from torch_geometric.utils import sort_edge_index


@pytest.mark.skipif(not torch_geometric.typing.WITH_METIS,
                    reason='Not compiled with METIS support')
def test_cluster_gcn():
    adj = torch.tensor([
        [1, 1, 1, 0, 1, 0],
        [1, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1],
    ])

    x = torch.Tensor([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
    edge_index = adj.nonzero(as_tuple=False).t()
    edge_attr = torch.arange(edge_index.size(1))
    n_id = torch.arange(6)
    data = Data(x=x, n_id=n_id, edge_index=edge_index, edge_attr=edge_attr)
    data.num_nodes = 6

    cluster_data = ClusterData(data, num_parts=2, log=False)

    partition = cluster_data._partition(
        edge_index, cluster=torch.tensor([0, 1, 0, 1, 0, 1]))
    assert partition.partptr.tolist() == [0, 3, 6]
    assert partition.node_perm.tolist() == [0, 2, 4, 1, 3, 5]
    assert partition.edge_perm.tolist() == [
        0, 2, 3, 1, 8, 9, 10, 14, 15, 16, 4, 5, 6, 7, 11, 12, 13, 17, 18, 19
    ]

    assert cluster_data.partition.partptr.tolist() == [0, 3, 6]
    assert torch.equal(
        cluster_data.partition.node_perm.sort()[0],
        torch.arange(data.num_nodes),
    )
    assert torch.equal(
        cluster_data.partition.edge_perm.sort()[0],
        torch.arange(data.num_edges),
    )

    out = cluster_data[0]
    expected = data.subgraph(out.n_id)
    out.validate()
    assert out.num_nodes == 3
    assert out.n_id.size() == (3, )
    assert torch.equal(out.x, expected.x)
    tmp = sort_edge_index(expected.edge_index, expected.edge_attr)
    assert torch.equal(out.edge_index, tmp[0])
    assert torch.equal(out.edge_attr, tmp[1])

    out = cluster_data[1]
    out.validate()
    assert out.num_nodes == 3
    assert out.n_id.size() == (3, )
    expected = data.subgraph(out.n_id)
    assert torch.equal(out.x, expected.x)
    tmp = sort_edge_index(expected.edge_index, expected.edge_attr)
    assert torch.equal(out.edge_index, tmp[0])
    assert torch.equal(out.edge_attr, tmp[1])

    loader = ClusterLoader(cluster_data, batch_size=1)
    iterator = iter(loader)

    out = next(iterator)
    out.validate()
    assert out.num_nodes == 3
    assert out.n_id.size() == (3, )
    expected = data.subgraph(out.n_id)
    assert torch.equal(out.x, expected.x)
    tmp = sort_edge_index(expected.edge_index, expected.edge_attr)
    assert torch.equal(out.edge_index, tmp[0])
    assert torch.equal(out.edge_attr, tmp[1])

    out = next(iterator)
    out.validate()
    assert out.num_nodes == 3
    assert out.n_id.size() == (3, )
    expected = data.subgraph(out.n_id)
    assert torch.equal(out.x, expected.x)
    tmp = sort_edge_index(expected.edge_index, expected.edge_attr)
    assert torch.equal(out.edge_index, tmp[0])
    assert torch.equal(out.edge_attr, tmp[1])

    loader = ClusterLoader(cluster_data, batch_size=2, shuffle=False)
    out = next(iter(loader))
    out.validate()
    assert out.num_nodes == 6
    assert out.n_id.size() == (6, )
    expected = data.subgraph(out.n_id)
    assert torch.equal(out.x, expected.x)
    tmp = sort_edge_index(expected.edge_index, expected.edge_attr)
    assert torch.equal(out.edge_index, tmp[0])
    assert torch.equal(out.edge_attr, tmp[1])


@pytest.mark.skipif(not torch_geometric.typing.WITH_METIS,
                    reason='Not compiled with METIS support')
def test_keep_inter_cluster_edges():
    adj = torch.tensor([
        [1, 1, 1, 0, 1, 0],
        [1, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1],
    ])

    x = torch.Tensor([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
    edge_index = adj.nonzero(as_tuple=False).t()
    edge_attr = torch.arange(edge_index.size(1))
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.num_nodes = 6

    cluster_data = ClusterData(data, num_parts=2, log=False,
                               keep_inter_cluster_edges=True)

    data = cluster_data[0]
    assert data.edge_index[0].min() == 0
    assert data.edge_index[0].max() == 2
    assert data.edge_index[1].min() == 0
    assert data.edge_index[1].max() > 2
    assert data.edge_index.size(1) == data.edge_attr.size(0)

    data = cluster_data[1]
    assert data.edge_index[0].min() == 0
    assert data.edge_index[0].max() == 2
    assert data.edge_index[1].min() == 0
    assert data.edge_index[1].max() > 2
    assert data.edge_index.size(1) == data.edge_attr.size(0)


@onlyFullTest
@pytest.mark.skipif(not torch_geometric.typing.WITH_METIS,
                    reason='Not compiled with METIS support')
def test_cluster_gcn_correctness(get_dataset):
    dataset = get_dataset('Cora')
    data = dataset[0].clone()
    data.n_id = torch.arange(data.num_nodes)
    cluster_data = ClusterData(data, num_parts=10, log=False)
    loader = ClusterLoader(cluster_data, batch_size=3, shuffle=False)

    for batch1 in loader:
        batch1.validate()
        batch2 = data.subgraph(batch1.n_id)
        assert batch1.num_nodes == batch2.num_nodes
        assert batch1.num_edges == batch2.num_edges
        assert torch.equal(batch1.x, batch2.x)
        assert torch.equal(
            batch1.edge_index,
            sort_edge_index(batch2.edge_index),
        )


if __name__ == '__main__':
    import argparse

    from ogb.nodeproppred import PygNodePropPredDataset
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()

    data = PygNodePropPredDataset('ogbn-products', root='/tmp/ogb')[0]

    loader = ClusterLoader(
        ClusterData(data, num_parts=15_000, save_dir='/tmp/ogb/ogbn_products'),
        batch_size=32,
        shuffle=True,
        num_workers=args.num_workers,
    )

    for batch in tqdm(loader):
        pass
