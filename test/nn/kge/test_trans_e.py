import torch

from torch_geometric.nn import TransE


def test_trans_e():
    model = TransE(num_nodes=10, num_relations=5, hidden_channels=32)
    assert str(model) == 'TransE(10, num_relations=5, hidden_channels=32)'

    edge_index = torch.tensor([[0, 2, 4, 6, 8], [1, 3, 5, 7, 9]])
    edge_type = torch.tensor([0, 1, 2, 3, 4])

    out = model(edge_index, edge_type)
    assert out.size() == (5, )
