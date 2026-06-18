import pytest
import torch

from torch_geometric.nn import Attri2Vec
from torch_geometric.testing import (
    has_package,
    is_full_test,
    withDevice,
    withPackage,
)


@withDevice
@withPackage('pyg_lib')
@pytest.mark.parametrize('mapping', ['linear', 'relu', 'sigmoid', 'kernel'])
def test_attri2vec(device, mapping):
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], device=device)
    x = torch.randn(3, 16, device=device)
    model = Attri2Vec(edge_index,
                      x,
                      embedding_dim=32,
                      walk_length=5,
                      context_size=3,
                      mapping=mapping).to(device)
    assert str(model) == 'Attri2Vec(3, 32)'
    assert model(torch.arange(3, device=device)).size() == (3, 32)

    pos_rw, neg_rw = model.sample(torch.arange(3))
    assert float(
        model.loss(pos_rw.to(device), neg_rw.to(device)).detach()
    ) >= 0

    if has_package('sklearn'):
        acc = model.test(torch.ones(20, 32), torch.randint(10, (20, )),
                         torch.ones(20, 32), torch.randint(10, (20, )))
        assert 0 <= acc <= 1

    if is_full_test():
        jit = torch.jit.script(model)
        assert jit(torch.arange(3, device=device)).size() == (3, 32)
        pos_rw, neg_rw = jit.sample(torch.arange(3))
        assert float(
            jit.loss(pos_rw.to(device), neg_rw.to(device)).detach()
        ) >= 0


@withPackage('pyg_lib')
def test_attri2vec_kernel_odd_dim():
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    x = torch.randn(3, 16)
    with pytest.raises(AssertionError):
        Attri2Vec(edge_index, x, embedding_dim=33, walk_length=5,
                  context_size=3, mapping='kernel')


@withPackage('pyg_lib')
def test_attri2vec_sparse():
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    x = torch.randn(3, 16)
    model = Attri2Vec(edge_index, x, embedding_dim=32, walk_length=5,
                      context_size=3, sparse=True)
    assert model.embedding.sparse
