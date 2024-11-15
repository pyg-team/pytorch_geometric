from torch_geometric.datasets import GitMolDataset
from torch_geometric.testing import withPackage


@withPackage('torchvision', 'rdkit', 'PIL')
def test_git_mol_dataset():
    dataset = GitMolDataset(root='./data/GITMol')

    assert len(dataset) == 3610
    assert dataset[0].image.size() == (1, 3, 224, 224)
    assert dataset[0].num_node_features == 9
    assert dataset[0].num_edge_features == 3
