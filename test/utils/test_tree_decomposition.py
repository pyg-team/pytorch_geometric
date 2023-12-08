import pytest

from torch_geometric.testing import withPackage
from torch_geometric.utils import tree_decomposition


@withPackage('rdkit')
@pytest.mark.parametrize('smiles', [
    r'F/C=C/F',
    r'C/C(=C\C(=O)c1ccc(C)o1)Nc1ccc2c(c1)OCO2',
])
def test_tree_decomposition(smiles):
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    tree_decomposition(mol)  # TODO Test output
