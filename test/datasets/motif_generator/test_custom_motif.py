import pytest
import torch

from torch_geometric.data import Data
from torch_geometric.datasets.motif_generator import CustomMotif
from torch_geometric.testing import withPackage


def test_custom_motif_pyg_data():
    structure = Data(
        num_nodes=3,
        edge_index=torch.tensor([[0, 1, 2, 1, 2, 0], [1, 2, 0, 0, 1, 2]]),
    )

    motif_generator = CustomMotif(structure)
    assert str(motif_generator) == 'CustomMotif()'

    assert structure == motif_generator()


@withPackage('networkx')
def test_custom_motif_networkx():
    import networkx as nx

    structure = nx.gnm_random_graph(5, 10, seed=2000)

    motif_generator = CustomMotif(structure)
    assert str(motif_generator) == 'CustomMotif()'

    out = motif_generator()
    assert len(out) == 2
    assert out.num_nodes == 5
    assert out.num_edges == 20


def test_custom_motif_unknown():
    with pytest.raises(ValueError, match="motif structure of type"):
        CustomMotif(structure='unknown')
