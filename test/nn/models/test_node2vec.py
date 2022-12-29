import torch

from torch_geometric.nn import Node2Vec
from torch_geometric.testing import is_full_test, withPackage


@withPackage('torch_cluster')
def test_node2vec():
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])

    model = Node2Vec(edge_index, embedding_dim=16, walk_length=2,
                     context_size=2)
    assert model.__repr__() == 'Node2Vec(3, 16)'

    assert model(torch.arange(3)).size() == (3, 16)

    pos_rw, neg_rw = model.sample(torch.arange(3))
    assert float(model.loss(pos_rw, neg_rw)) >= 0

    acc = model.test(torch.ones(20, 16), torch.randint(10, (20, )),
                     torch.ones(20, 16), torch.randint(10, (20, )))
    assert 0 <= acc and acc <= 1

    if is_full_test():
        jit = torch.jit.export(model)

        assert jit(torch.arange(3)).size() == (3, 16)

        pos_rw, neg_rw = jit.sample(torch.arange(3))
        assert float(jit.loss(pos_rw, neg_rw)) >= 0
