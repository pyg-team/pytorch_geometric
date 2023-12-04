from torch_geometric.datasets.motif_generator import CycleMotif


def test_cycle_motif():
    motif_generator = CycleMotif(5)
    assert str(motif_generator) == 'CycleMotif(5)'

    motif = motif_generator()
    assert len(motif) == 2
    assert motif.num_nodes == 5
    assert motif.num_edges == 10
    assert motif.edge_index.tolist() == [
        [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
        [1, 4, 0, 2, 1, 3, 2, 4, 0, 3],
    ]
