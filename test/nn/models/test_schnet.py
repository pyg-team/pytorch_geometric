import pytest
import torch

from torch_geometric.data import Batch, Data
from torch_geometric.nn import SchNet
from torch_geometric.nn.models.schnet import RadiusInteractionGraph
from torch_geometric.testing import is_full_test, withPackage


def generate_data():
    return Data(
        z=torch.randint(1, 10, (20, )),
        pos=torch.randn(20, 3),
    )


@withPackage('torch_cluster')
@withPackage('ase')
@pytest.mark.parametrize('use_interaction_graph', [False, True])
@pytest.mark.parametrize('use_atomref', [False, True])
def test_schnet(use_interaction_graph, use_atomref):
    data = generate_data()

    interaction_graph = None
    if use_interaction_graph:
        interaction_graph = RadiusInteractionGraph(cutoff=6.0)

    model = SchNet(
        hidden_channels=16,
        num_filters=16,
        num_interactions=2,
        interaction_graph=interaction_graph,
        num_gaussians=10,
        cutoff=6.0,
        dipole=True,
        atomref=torch.randn(100, 1) if use_atomref else None,
    )

    assert str(model) == ('SchNet(hidden_channels=16, num_filters=16, '
                          'num_interactions=2, num_gaussians=10, cutoff=6.0)')

    with torch.no_grad():
        out = model(data.z, data.pos)
        assert out.size() == (1, 1)

        if is_full_test():
            jit = torch.jit.export(model)
            out = jit(data.z, data.pos)
            assert out.size() == (1, 1)


@withPackage('torch_cluster')
def test_schnet_batch():
    num_graphs = 3
    batch = [generate_data() for _ in range(num_graphs)]
    batch = Batch.from_data_list(batch)

    model = SchNet(
        hidden_channels=16,
        num_filters=16,
        num_interactions=2,
        num_gaussians=10,
        cutoff=6.0,
    )

    with torch.no_grad():
        out = model(batch.z, batch.pos, batch.batch)
        assert out.size() == (num_graphs, 1)
