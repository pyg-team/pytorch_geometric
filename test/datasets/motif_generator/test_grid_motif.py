from torch_geometric.datasets.motif_generator import GridMotif


def test_grid_motif():
    motif_generator = GridMotif()
    assert str(motif_generator) == 'GridMotif()'

    motif = motif_generator()
    assert len(motif) == 3
    assert motif.num_nodes == 9
    assert motif.num_edges == 24
    assert motif.edge_index.size() == (2, 24)
    assert motif.edge_index.min() == 0
    assert motif.edge_index.max() == 8
    assert motif.y.size() == (9, )
    assert motif.y.min() == 0
    assert motif.y.max() == 2
