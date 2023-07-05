import torch

from torch_geometric.nn import GCN, DeepGraphInfomax
from torch_geometric.testing import is_full_test, withCUDA


@withCUDA
def test_infomax(device):
    def corruption(z):
        return z + 1

    model = DeepGraphInfomax(
        hidden_channels=16,
        encoder=lambda x: x,
        summary=lambda z, *args: z.mean(dim=0),
        corruption=lambda x: x + 1,
    ).to(device)
    assert str(model) == 'DeepGraphInfomax(16)'

    x = torch.ones(20, 16, device=device)

    pos_z, neg_z, summary = model(x)
    assert pos_z.size() == (20, 16)
    assert neg_z.size() == (20, 16)
    assert summary.size() == (16, )

    loss = model.loss(pos_z, neg_z, summary)
    assert float(loss) >= 0

    if is_full_test():
        jit = torch.jit.export(model)
        pos_z, neg_z, summary = jit(x)
        assert pos_z.size() == (20, 16) and neg_z.size() == (20, 16)
        assert summary.size() == (16, )

    acc = model.test(
        train_z=torch.ones(20, 16),
        train_y=torch.randint(10, (20, )),
        test_z=torch.ones(20, 16),
        test_y=torch.randint(10, (20, )),
    )
    assert 0 <= acc <= 1


@withCUDA
def test_infomax_predefined_model(device):
    def corruption(x, edge_index, edge_weight):
        return (
            x[torch.randperm(x.size(0), device=x.device)],
            edge_index,
            edge_weight,
        )

    model = DeepGraphInfomax(
        hidden_channels=16,
        encoder=GCN(16, 16, num_layers=2),
        summary=lambda z, *args, **kwargs: z.mean(dim=0).sigmoid(),
        corruption=corruption,
    ).to(device)

    x = torch.randn(4, 16, device=device)
    edge_index = torch.tensor(
        [[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]],
        device=device,
    )
    edge_weight = torch.rand(edge_index.size(1), device=device)

    pos_z, neg_z, summary = model(x, edge_index, edge_weight=edge_weight)
    assert pos_z.size() == (4, 16)
    assert neg_z.size() == (4, 16)
    assert summary.size() == (16, )

    loss = model.loss(pos_z, neg_z, summary)
    assert float(loss) >= 0
