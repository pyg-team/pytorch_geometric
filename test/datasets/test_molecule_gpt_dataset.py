from torch_geometric.datasets import MoleculeGPTDataset
from torch_geometric.testing import onlyOnline, withPackage


@onlyOnline
@withPackage('transformers', 'sentencepiece', 'accelerate', 'rdkit')
def test_molecule_gpt_dataset():
    dataset = MoleculeGPTDataset(root='./data/MoleculeGPT')
    assert str(dataset) == f'MoleculeGPTDataset({len(dataset)})'
    assert dataset.num_edge_features == 4
    assert dataset.num_node_features == 6
