import torch
from torch_geometric.data import Data, ClusterData, ClusterLoader
from torch_geometric.utils import to_dense_adj


def test_cluster_data():
    adj = torch.tensor([
        [1, 1, 1, 0, 1, 0],
        [1, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1],
    ])

    edge_index = adj.nonzero().t()
    x = torch.Tensor([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
    data = Data(edge_index=edge_index, x=x, num_nodes=6)

    cluster_data = ClusterData(data, num_parts=2)

    assert cluster_data.partptr.tolist() == [0, 3, 6]
    assert cluster_data.perm.tolist() == [0, 2, 4, 1, 3, 5]
    assert cluster_data.data.x.tolist() == [
        [0, 0],
        [2, 2],
        [4, 4],
        [1, 1],
        [3, 3],
        [5, 5],
    ]
    assert cluster_data.data.adj.to_dense().tolist() == [
        [1, 1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [1, 0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1, 1],
    ]

    loader = ClusterLoader(cluster_data, batch_size=1)
    it = iter(loader)

    data = next(it)
    assert data.x.tolist() == [[0, 0], [2, 2], [4, 4]]
    assert data.edge_index.tolist() == [[0, 0, 0, 1, 1, 1, 2, 2, 2],
                                        [0, 1, 2, 0, 1, 2, 0, 1, 2]]

    data = next(it)
    assert data.x.tolist() == [[1, 1], [3, 3], [5, 5]]
    assert data.edge_index.tolist() == [[0, 0, 0, 1, 1, 1, 2, 2, 2],
                                        [0, 1, 2, 0, 1, 2, 0, 1, 2]]

    torch.manual_seed(1)
    loader = ClusterLoader(cluster_data, batch_size=2, shuffle=True)
    data = next(iter(loader))
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

    torch.manual_seed(2)
    loader = ClusterLoader(cluster_data, batch_size=2, shuffle=True)
    data = next(iter(loader))
    assert data.x.tolist() == [
        [1, 1],
        [3, 3],
        [5, 5],
        [0, 0],
        [2, 2],
        [4, 4],
    ]
    assert to_dense_adj(data.edge_index).squeeze().tolist() == [
        [1, 1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [1, 0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1, 1],
    ]
