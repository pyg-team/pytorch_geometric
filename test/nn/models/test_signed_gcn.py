import torch

from torch_geometric.nn import SignedGCN
from torch_geometric.testing import has_package, is_full_test


# @withPackage('sklearn')
def test_signed_gcn():
    model = SignedGCN(8, 16, num_layers=2, lamb=5)
    assert str(model) == 'SignedGCN(8, 16, num_layers=2)'
    N, E = 20, 40
    all_index = torch.randperm(N*N, dtype=torch.long)
    all_index = torch.stack([all_index // N, all_index % N], dim=0)
    pos_index = all_index[:, :E]
    neg_index = all_index[:, E:2*E]

    train_pos_index, test_pos_index = model.split_edges(pos_index)
    train_neg_index, test_neg_index = model.split_edges(neg_index)

    assert train_pos_index.size() == (2, 32)
    assert test_pos_index.size() == (2, 8)
    assert train_neg_index.size() == (2, 32)
    assert test_neg_index.size() == (2, 8)

    if has_package('sklearn'):
        x = model.create_spectral_features(
            train_pos_index,
            train_neg_index,
            num_nodes=N,
        )
        assert x.size() == (N, 8)
    else:
        x = torch.randn(N, 8)

    z = model(x, train_pos_index, train_neg_index)
    assert z.size() == (N, 16)

    loss = model.loss(z, train_pos_index, train_neg_index)
    assert loss.item() >= 0

    if has_package('sklearn'):
        auc, f1 = model.test(z, test_pos_index, test_neg_index)
        assert auc >= 0
        assert f1 >= 0

    if is_full_test():
        jit = torch.jit.export(model)
        assert torch.allclose(jit(x, train_pos_index, train_neg_index), z)
