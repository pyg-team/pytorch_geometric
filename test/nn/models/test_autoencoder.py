import torch
from torch import Tensor as T

from torch_geometric.data import Data
from torch_geometric.nn import ARGA, ARGVA, GAE, VGAE
from torch_geometric.testing import is_full_test
from torch_geometric.transforms import RandomLinkSplit


def test_gae():
    model = GAE(encoder=lambda x: x)
    model.reset_parameters()

    x = torch.tensor([[1.0, -1.0], [1.0, 2.0], [2.0, 1.0]])
    z = model.encode(x)
    assert torch.allclose(z, x)

    adj = model.decoder.forward_all(z)
    expected = torch.tensor([
        [2.0, -1.0, 1.0],
        [-1.0, 5.0, 4.0],
        [1.0, 4.0, 5.0],
    ]).sigmoid()
    assert torch.allclose(adj, expected)

    edge_index = torch.tensor([[0, 1], [1, 2]])
    value = model.decode(z, edge_index)
    assert torch.allclose(value, torch.tensor([-1.0, 4.0]).sigmoid())

    if is_full_test():
        jit = torch.jit.export(model)
        assert torch.allclose(jit.encode(x), z)
        assert torch.allclose(jit.decode(z, edge_index), value)

    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                               [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    data = Data(edge_index=edge_index, num_nodes=11)
    transform = RandomLinkSplit(split_labels=True,
                                add_negative_train_samples=False)
    train_data, val_data, test_data = transform(data)

    z = torch.randn(11, 16)
    loss = model.recon_loss(z, train_data.pos_edge_label_index)
    assert float(loss) > 0

    auc, ap = model.test(z, val_data.pos_edge_label_index,
                         val_data.neg_edge_label_index)
    assert auc >= 0 and auc <= 1 and ap >= 0 and ap <= 1


def test_vgae():
    model = VGAE(encoder=lambda x: (x, x))

    x = torch.tensor([[1.0, -1.0], [1.0, 2.0], [2.0, 1.0]])
    model.encode(x)
    assert float(model.kl_loss()) > 0

    model.eval()
    model.encode(x)

    if is_full_test():
        jit = torch.jit.export(model)
        jit.encode(x)
        assert float(jit.kl_loss()) > 0


def test_arga():
    model = ARGA(encoder=lambda x: x, discriminator=lambda x: T([0.5]))
    model.reset_parameters()

    x = torch.tensor([[1.0, -1.0], [1.0, 2.0], [2.0, 1.0]])
    z = model.encode(x)

    assert float(model.reg_loss(z)) > 0
    assert float(model.discriminator_loss(z)) > 0

    if is_full_test():
        jit = torch.jit.export(model)
        assert torch.allclose(jit.encode(x), z)
        assert float(jit.reg_loss(z)) > 0
        assert float(jit.discriminator_loss(z)) > 0


def test_argva():
    model = ARGVA(encoder=lambda x: (x, x), discriminator=lambda x: T([0.5]))

    x = torch.tensor([[1.0, -1.0], [1.0, 2.0], [2.0, 1.0]])
    model.encode(x)
    model.reparametrize(model.__mu__, model.__logstd__)
    assert float(model.kl_loss()) > 0

    if is_full_test():
        jit = torch.jit.export(model)
        jit.encode(x)
        jit.reparametrize(jit.__mu__, jit.__logstd__)
        assert float(jit.kl_loss()) > 0


def test_init():
    encoder = torch.nn.Linear(16, 32)
    decoder = torch.nn.Linear(32, 16)
    discriminator = torch.nn.Linear(32, 1)

    GAE(encoder, decoder)
    VGAE(encoder, decoder)
    ARGA(encoder, discriminator, decoder)
    ARGVA(encoder, discriminator, decoder)
