import pytest

from torch_geometric.testing import withPackage
from torch_geometric.utils import from_rdmol, from_smiles, to_rdmol, to_smiles

smiles = [
    r'F/C=C/F',
    r'F/C=C\F',
    r'F/C=C\F',
    (r'COc1cccc([C@@H]2Oc3ccc(OC)cc3/C(=N/OC[C@@H](C)[C@H](OCc3ccccc3)'
     r'C(C)C)[C@@H]2O)c1'),
    r'C/C(=C\C(=O)c1ccc(C)o1)Nc1ccc2c(c1)OCO2',
    r'F[B-](F)(F)c1cnc2ccccc2c1',
    r'COC(=O)[C@@]1(Cc2ccccc2)[C@H]2C(=O)N(C)C(=O)[C@H]2[C@H]2CN=C(SC)N21',
    (r'O=C(O)c1ccc(NS(=O)(=O)c2ccc3c(c2)C(=O)c2cc(S(=O)(=O)Nc4ccc(C(=O)O)'
     r'cc4)ccc2-3)cc1'),
]


@withPackage('rdkit')
@pytest.mark.parametrize('smiles', smiles)
def test_from_to_smiles(smiles):
    data = from_smiles(smiles)
    assert to_smiles(data) == smiles


@withPackage('rdkit')
@pytest.mark.parametrize('smiles', smiles)
def test_from_to_rdmol(smiles):
    from rdkit import Chem
    mol1 = Chem.MolFromSmiles(smiles)
    data = from_rdmol(mol1)
    mol2 = to_rdmol(data)
    assert Chem.MolToSmiles(mol1) == Chem.MolToSmiles(mol2)
