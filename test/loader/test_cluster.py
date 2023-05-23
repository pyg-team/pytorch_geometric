import pytest
import torch

import torch_geometric.typing
from torch_geometric.data import Data
from torch_geometric.loader import ClusterData, ClusterLoader
from torch_geometric.testing import onlyFullTest
from torch_geometric.utils import to_dense_adj

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
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.num_nodes = 6

    cluster_data = ClusterData(data, num_parts=2, log=False)

    assert cluster_data.partition.rowptr.tolist() == [0, 4, 7, 10, 14, 17, 20]
    assert cluster_data.partition.col.tolist() == [
        0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 0, 3, 4, 5, 3, 4, 5, 3, 4, 5
    ]
    assert cluster_data.partition.partptr.tolist() == [0, 3, 6]
    assert cluster_data.partition.node_perm.tolist() == [0, 2, 4, 1, 3, 5]
    assert cluster_data.partition.edge_perm.tolist() == [
        0, 2, 3, 1, 8, 9, 10, 14, 15, 16, 4, 5, 6, 7, 11, 12, 13, 17, 18, 19
    ]
    assert cluster_data.data.x.tolist() == [
        [0, 0],
        [2, 2],
        [4, 4],
        [1, 1],
        [3, 3],
        [5, 5],
    ]

    data = cluster_data[0]
    assert data.num_nodes == 3
    assert data.x.tolist() == [[0, 0], [2, 2], [4, 4]]
    assert data.edge_index.tolist() == [[0, 0, 0, 1, 1, 1, 2, 2, 2],
                                        [0, 1, 2, 0, 1, 2, 0, 1, 2]]
    assert data.edge_attr.tolist() == [0, 2, 3, 8, 9, 10, 14, 15, 16]

    data = cluster_data[1]
    assert data.num_nodes == 3
    assert data.x.tolist() == [[1, 1], [3, 3], [5, 5]]
    assert data.edge_index.tolist() == [[0, 0, 0, 1, 1, 1, 2, 2, 2],
                                        [0, 1, 2, 0, 1, 2, 0, 1, 2]]
    assert data.edge_attr.tolist() == [5, 6, 7, 11, 12, 13, 17, 18, 19]

    loader = ClusterLoader(cluster_data, batch_size=1)
    iterator = iter(loader)

    data = next(iterator)
    assert data.x.tolist() == [[0, 0], [2, 2], [4, 4]]
    assert data.edge_index.tolist() == [[0, 0, 0, 1, 1, 1, 2, 2, 2],
                                        [0, 1, 2, 0, 1, 2, 0, 1, 2]]
    assert data.edge_attr.tolist() == [0, 2, 3, 8, 9, 10, 14, 15, 16]

    data = next(iterator)
    assert data.x.tolist() == [[1, 1], [3, 3], [5, 5]]
    assert data.edge_index.tolist() == [[0, 0, 0, 1, 1, 1, 2, 2, 2],
                                        [0, 1, 2, 0, 1, 2, 0, 1, 2]]
    assert data.edge_attr.tolist() == [5, 6, 7, 11, 12, 13, 17, 18, 19]

    loader = ClusterLoader(cluster_data, batch_size=2, shuffle=False)
    data = next(iter(loader))
    assert data.num_nodes == 6
    assert data.x.tolist() == [
        [0, 0],
        [2, 2],
        [4, 4],
        [1, 1],
        [3, 3],
        [5, 5],
    ]
    assert to_dense_adj(data.edge_index).squeeze().tolist() == [
        [1, 1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [1, 0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1, 1],
    ]


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
