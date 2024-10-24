from torch_geometric.datasets import GitMolDataset


def test_git_mol_dataset():
    dataset = GitMolDataset(root='./data/GITMol')

    assert len(dataset) == 3610
    assert dataset[0].image.size() == (3, 224, 224)
    assert dataset[0].num_node_features == 9
    assert dataset[0].num_edge_features == 3
