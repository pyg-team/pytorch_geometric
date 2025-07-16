import pytest
import torch

import torch_geometric.typing
from torch_geometric.nn import Node2Vec
from torch_geometric.testing import (
    has_package,
    is_full_test,
    withDevice,
    withPackage,
)


@withDevice
@withPackage('pyg_lib|torch_cluster')
@pytest.mark.parametrize('p', [1.0])
@pytest.mark.parametrize('q', [1.0, 0.5])
def test_node2vec(device, p, q):
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], device=device)
    kwargs = dict(embedding_dim=16, walk_length=2, context_size=2, p=p, q=q)

    if not torch_geometric.typing.WITH_TORCH_CLUSTER and q != 1.0:
        with pytest.raises(ImportError, match="requires the 'torch-cluster'"):
            model = Node2Vec(edge_index, **kwargs)
        return

    model = Node2Vec(edge_index, **kwargs).to(device)
    assert str(model) == 'Node2Vec(3, 16)'

    assert model(torch.arange(3, device=device)).size() == (3, 16)

    pos_rw, neg_rw = model.sample(torch.arange(3))
    assert float(model.loss(pos_rw.to(device), neg_rw.to(device))) >= 0

    if has_package('sklearn'):
        acc = model.test(torch.ones(20, 16), torch.randint(10, (20, )),
                         torch.ones(20, 16), torch.randint(10, (20, )))
        assert 0 <= acc and acc <= 1

    if is_full_test():
        jit = torch.jit.script(model)

        assert jit(torch.arange(3, device=device)).size() == (3, 16)

        pos_rw, neg_rw = jit.sample(torch.arange(3))
        assert float(jit.loss(pos_rw.to(device), neg_rw.to(device))) >= 0
