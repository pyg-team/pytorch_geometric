import torch

from torch_geometric.nn import DeepGraphInfomax
from torch_geometric.testing import is_full_test


def test_deep_graph_infomax():
    def corruption(z):
        return z + 1

    model = DeepGraphInfomax(hidden_channels=16, encoder=lambda x: x,
                             summary=lambda z, *args: z.mean(dim=0),
                             corruption=lambda x: x + 1)

    assert model.__repr__() == 'DeepGraphInfomax(16)'

    x = torch.ones(20, 16)

    pos_z, neg_z, summary = model(x)
    assert pos_z.size() == (20, 16) and neg_z.size() == (20, 16)
    assert summary.size() == (16, )

    if is_full_test():
        jit = torch.jit.export(model)
        pos_z, neg_z, summary = jit(x)
        assert pos_z.size() == (20, 16) and neg_z.size() == (20, 16)
        assert summary.size() == (16, )

    loss = model.loss(pos_z, neg_z, summary)
    assert 0 <= loss.item()

    acc = model.test(torch.ones(20, 16), torch.randint(10, (20, )),
                     torch.ones(20, 16), torch.randint(10, (20, )))
    assert 0 <= acc and acc <= 1
