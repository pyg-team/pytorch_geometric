import pytest
import torch

import torch_geometric.typing
from torch_geometric.data import Data
from torch_geometric.loader import ClusterData, ClusterLoader
from torch_geometric.testing import onlyFullTest
from torch_geometric.utils import sort_edge_index

try:
    rowptr = torch.tensor([0, 1])
    col = torch.tensor([0])
    torch.ops.torch_sparse.partition(rowptr, col, None, 1, True)
    WITH_METIS = True
except (AttributeError, RuntimeError):
    WITH_METIS = False or torch_geometric.typing.WITH_METIS


@pytest.mark.skipif(not WITH_METIS, reason='Not compiled with METIS support')
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

    assert cluster_data.partition.partptr.tolist() == [0, 3, 6]
    assert cluster_data.partition.node_perm.min() == 0
    assert cluster_data.partition.node_perm.max() == 5
    assert cluster_data.partition.node_perm.size() == (6, )
    assert cluster_data.partition.node_perm.unique().numel() == 6

    out = cluster_data[0]
    out.validate()
    assert out.num_nodes == 3
    assert out.n_id.size() == (3, )
    assert out.x.size() == (3, 2)
    assert torch.equal(out.x, data.subgraph(out.n_id).x)
    assert torch.equal(out.edge_index, data.subgraph(out.n_id).edge_index)
    assert torch.equal(out.edge_attr, data.subgraph(out.n_id).edge_attr)

    out = cluster_data[1]
    out.validate()
    assert out.num_nodes == 3
    assert out.n_id.size() == (3, )
    assert out.x.size() == (3, 2)
    assert torch.equal(out.x, data.subgraph(out.n_id).x)
    assert torch.equal(out.edge_index, data.subgraph(out.n_id).edge_index)
    assert torch.equal(out.edge_attr, data.subgraph(out.n_id).edge_attr)

    loader = ClusterLoader(cluster_data, batch_size=1)
    iterator = iter(loader)

    out = next(iterator)
    out.validate()
    assert out.num_nodes == 3
    assert out.n_id.size() == (3, )
    assert out.x.size() == (3, 2)
    assert torch.equal(out.x, data.subgraph(out.n_id).x)
    assert torch.equal(out.edge_index, data.subgraph(out.n_id).edge_index)
    assert torch.equal(out.edge_attr, data.subgraph(out.n_id).edge_attr)

    out = next(iterator)
    out.validate()
    assert out.num_nodes == 3
    assert out.n_id.size() == (3, )
    assert out.x.size() == (3, 2)
    assert torch.equal(out.x, data.subgraph(out.n_id).x)
    assert torch.equal(out.edge_index, data.subgraph(out.n_id).edge_index)
    assert torch.equal(out.edge_attr, data.subgraph(out.n_id).edge_attr)

    loader = ClusterLoader(cluster_data, batch_size=2, shuffle=False)
    out = next(iter(loader))
    assert out.num_nodes == 6
    assert out.n_id.size() == (6, )
    assert out.x.size() == (6, 2)
    assert torch.equal(out.x, data.subgraph(out.n_id).x)
    assert torch.equal(out.edge_index,
                       sort_edge_index(data.subgraph(out.n_id).edge_index))
    assert torch.equal(out.edge_attr, data.subgraph(out.n_id).edge_attr)


@pytest.mark.skipif(not WITH_METIS, reason='Not compiled with METIS support')
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
@pytest.mark.skipif(not WITH_METIS, reason='Not compiled with METIS support')
def test_cluster_gcn_correctness(get_dataset):
    dataset = get_dataset('Cora')
    data = dataset[0].clone()
    data.n_id = torch.arange(data.num_nodes)
    cluster_data = ClusterData(data, num_parts=10, log=False)
    loader = ClusterLoader(cluster_data, batch_size=3, shuffle=False)

    for batch1 in loader:
        batch2 = data.subgraph(batch1.n_id)
        assert batch1.num_nodes == batch2.num_nodes
        assert batch1.num_edges == batch2.num_edges


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
