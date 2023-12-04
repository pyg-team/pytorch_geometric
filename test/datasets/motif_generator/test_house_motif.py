from torch_geometric.datasets.motif_generator import HouseMotif


def test_house_motif():
    motif_generator = HouseMotif()
    assert str(motif_generator) == 'HouseMotif()'

    motif = motif_generator()
    assert len(motif) == 3
    assert motif.num_nodes == 5
    assert motif.num_edges == 12
    assert motif.y.min() == 0 and motif.y.max() == 2
