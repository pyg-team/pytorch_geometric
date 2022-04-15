import torch

from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import RemoveIsolatedNodes


def test_remove_isolated_nodes():
    assert RemoveIsolatedNodes().__repr__() == 'RemoveIsolatedNodes()'

    edge_index = torch.tensor([[0, 2, 1, 0], [2, 0, 1, 0]])
    edge_attr = torch.tensor([1, 2, 3, 4])
    x = torch.tensor([[1], [2], [3]])
    data = Data(edge_index=edge_index, edge_attr=edge_attr, x=x)
    data = RemoveIsolatedNodes()(data)
    assert len(data) == 3
    assert data.edge_index.tolist() == [[0, 1, 0], [1, 0, 0]]
    assert data.edge_attr.tolist() == [1, 2, 4]
    assert data.x.tolist() == [[1], [3]]

    data = HeteroData()
    data['paper'].x = torch.arange(6).type(torch.float)
    data['author'].x = torch.arange(6).type(torch.float)

    # first self loop to be isolated, second not, and third not self loop
    data['paper', 'paper'].edge_index = torch.Tensor([[0, 1, 2],
                                                      [0, 1,
                                                       3]]).type(torch.long)

    # remove isolation of first paper self loop, and add another
    data['paper', 'author'].edge_index = torch.Tensor([[1, 3, 5],
                                                       [0, 1,
                                                        2]]).type(torch.long)

    # only edge_attr on one type
    data['paper', 'author'].edge_attr = torch.Tensor([1, 2,
                                                      3]).type(torch.long)

    # add a duplicate (in node index) edge
    data['paper', 'cites',
         'author'].edge_index = torch.Tensor([[5], [2]]).type(torch.long)

    data = RemoveIsolatedNodes()(data)
    assert data['paper'].num_nodes == 4
    assert data['author'].num_nodes == 3
    assert torch.allclose(data['paper'].x, torch.Tensor([1, 2, 3, 5]))
