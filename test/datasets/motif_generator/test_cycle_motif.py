from torch_geometric.datasets.motif_generator import CycleMotif


def test_cycle_motif():
    n = 5
    motif_generator = CycleMotif(n)
    assert str(motif_generator) == 'CycleMotif()'

    motif = motif_generator()
    assert len(motif) == 2
    assert motif.num_nodes == n
    assert motif.num_edges == n
