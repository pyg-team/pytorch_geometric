from torch_geometric.datasets import MoleculeGPTDataset


# @onlyOnline
def test_molecule_gpt_dataset():
    dataset = MoleculeGPTDataset(root='./data/MoleculeGPT', force_reload=True)
    assert str(dataset) == f'MoleculeGPTDataset({len(dataset)})'
    assert dataset.num_edge_features == 4
    assert dataset.num_node_features == 5
