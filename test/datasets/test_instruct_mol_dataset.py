from torch_geometric.datasets import InstructMolDataset
from torch_geometric.testing import onlyFullTest, withPackage


@onlyFullTest
@withPackage('rdkit')
def test_instruct_mol_dataset():
    dataset = InstructMolDataset(root='./data/InstructMol')
    assert len(dataset) == 326689
    assert dataset.num_edge_features == 4
    assert dataset.num_node_features == 6
