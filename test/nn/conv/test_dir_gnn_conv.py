import pytest
import torch

from torch_geometric.nn import DirGNNConv, SAGEConv


def test_dir_gnn_conv():
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])

    conv = DirGNNConv(SAGEConv(16, 32))
    assert str(conv) == 'DirGNNConv(SAGEConv(16, 32, aggr=mean), alpha=0.5)'

    out = conv(x, edge_index)
    assert out.size() == (4, 32)


def test_static_dir_gnn_conv():
    x = torch.randn(3, 4, 16)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])

    conv = DirGNNConv(SAGEConv(16, 32))

    out = conv(x, edge_index)
    assert out.size() == (3, 4, 32)


def test_dir_gnn_conv_trainable_alpha():
    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])

    # Test non-trainable alpha
    conv_fixed_alpha = DirGNNConv(SAGEConv(16, 32), alpha=0.3,
                                  trainable_alpha=False)
    assert not conv_fixed_alpha.alpha.requires_grad
    optimizer = torch.optim.SGD(conv_fixed_alpha.parameters(), lr=0.01)
    out = conv_fixed_alpha(x, edge_index)
    loss = out.sum()
    loss.backward()
    old_alpha = conv_fixed_alpha.alpha.item()
    optimizer.step()
    new_alpha = conv_fixed_alpha.alpha.item()
    assert old_alpha == pytest.approx(new_alpha)

    # Test trainable alpha
    conv_trainable_alpha = DirGNNConv(SAGEConv(16, 32), alpha=0.3,
                                      trainable_alpha=True)
    assert conv_trainable_alpha.alpha.requires_grad
    optimizer = torch.optim.SGD(conv_trainable_alpha.parameters(), lr=0.01)
    out = conv_trainable_alpha(x, edge_index)
    loss = out.sum()
    loss.backward()
    old_alpha = conv_trainable_alpha.alpha.item()
    optimizer.step()
    new_alpha = conv_trainable_alpha.alpha.item()
    assert not old_alpha == pytest.approx(new_alpha)
