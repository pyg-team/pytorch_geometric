import torch
from torch_geometric.nn import Node2Vec


def test_node2vec():
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])

    model = Node2Vec(num_nodes=3, embedding_dim=16, walk_length=2,
                     context_size=2)
    assert model.__repr__() == 'Node2Vec(3, 16, p=1, q=1)'

    z = model(torch.arange(3))
    assert z.size() == (3, 16)

    loss = model.loss(edge_index)
    assert 0 <= loss.item()

    acc = model.test(
        torch.ones(20, 16), torch.randint(10, (20, )), torch.ones(20, 16),
        torch.randint(10, (20, )))
    assert 0 <= acc and acc <= 1
