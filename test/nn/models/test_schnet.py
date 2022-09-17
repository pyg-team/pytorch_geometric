import torch
import torch.nn.functional as F

from torch_geometric import seed_everything
from torch_geometric.data import Batch, Data
from torch_geometric.nn import SchNet
from torch_geometric.testing import onlyFullTest, withPackage
from torch_geometric.transforms import RadiusGraph


def generate_data():
    return Data(
        z=torch.randint(1, 10, (20, )),
        pos=torch.randn(20, 3),
        y=torch.tensor([1.]),
    )


@onlyFullTest
def test_schnet():
    seed_everything(0)
    data = generate_data()

    model = SchNet(hidden_channels=16, num_filters=16, num_interactions=2,
                   num_gaussians=10, cutoff=6.0)

    with torch.no_grad():
        out = model(data.z, data.pos)
        assert out.size() == (1, 1)

    num_steps = 10
    losses = torch.empty(num_steps)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for i in range(num_steps):
        optimizer.zero_grad()
        out = model(data.z, data.pos)
        loss = F.mse_loss(out.view(-1), data.y)
        loss.backward()
        optimizer.step()
        losses[i] = loss.detach()
    assert losses[num_steps - 1] < losses[0]


@withPackage('torch_cluster')
def test_schnet_batch():
    seed_everything(0)
    num_graphs = 3
    batch = [generate_data() for _ in range(num_graphs)]
    batch = Batch.from_data_list(batch)

    model = SchNet(hidden_channels=16, num_filters=16, num_interactions=2,
                   num_gaussians=10, cutoff=6.0)

    with torch.no_grad():
        out = model(batch.z, batch.pos, batch.batch)
        assert out.size() == (num_graphs, 1)


@withPackage('torch_cluster')
def test_schnet_batch_pre_compute():
    seed_everything(0)
    num_graphs = 3
    cutoff = 6.0

    transform = RadiusGraph(cutoff)

    batch = [transform(generate_data()) for _ in range(num_graphs)]
    batch = Batch.from_data_list(batch)

    model = SchNet(hidden_channels=16, num_filters=16, num_interactions=2,
                   num_gaussians=10, cutoff=cutoff)

    with torch.no_grad():
        out = model(batch.z, batch.pos, batch.batch, batch.edge_index)
        assert out.size() == (num_graphs, 1)
