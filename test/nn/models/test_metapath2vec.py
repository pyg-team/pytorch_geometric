import torch

from torch_geometric.nn import MetaPath2Vec
from torch_geometric.testing import withCUDA


@withCUDA
def test_metapath2vec(device):
    edge_index_dict = {
        ('author', 'writes', 'paper'):
        torch.tensor([[0, 1, 1, 2], [0, 0, 1, 1]], device=device),
        ('paper', 'written_by', 'author'):
        torch.tensor([[0, 0, 1, 1], [0, 1, 1, 2]], device=device)
    }

    metapath = [
        ('author', 'writes', 'paper'),
        ('paper', 'written_by', 'author'),
    ]

    model = MetaPath2Vec(edge_index_dict, embedding_dim=16, metapath=metapath,
                         walk_length=2, context_size=2).to(device)
    assert str(model) == 'MetaPath2Vec(5, 16)'

    z = model('author')
    assert z.size() == (3, 16)

    z = model('paper')
    assert z.size() == (2, 16)

    z = model('author', torch.arange(2, device=device))
    assert z.size() == (2, 16)

    pos_rw, neg_rw = model._sample(torch.arange(3))

    loss = model.loss(pos_rw.to(device), neg_rw.to(device))
    assert 0 <= loss.item()

    acc = model.test(torch.ones(20, 16), torch.randint(10, (20, )),
                     torch.ones(20, 16), torch.randint(10, (20, )))
    assert 0 <= acc and acc <= 1
