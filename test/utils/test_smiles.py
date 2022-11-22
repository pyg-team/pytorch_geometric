import pytest

from torch_geometric.testing import withPackage
from torch_geometric.utils import smiles


@withPackage('rdkit')
@pytest.mark.parametrize('test_smile', [
    'F/C=C/F', 'F/C=C\F', 'F/C=C\F',
    'COc1cccc([C@@H]2Oc3ccc(OC)cc3/C(=N/OC[C@@H](C)[C@H](OCc3ccccc3)C(C)C)[C@@H]2O)c1',
    'C/C(=C\C(=O)c1ccc(C)o1)Nc1ccc2c(c1)OCO2', 'F[B-](F)(F)c1cnc2ccccc2c1',
    'COC(=O)[C@@]1(Cc2ccccc2)[C@H]2C(=O)N(C)C(=O)[C@H]2[C@H]2CN=C(SC)N21',
    'O=C(O)c1ccc(NS(=O)(=O)c2ccc3c(c2)C(=O)c2cc(S(=O)(=O)Nc4ccc(C(=O)O)cc4)ccc2-3)cc1'
])
def test_from_to_smiles(test_smile):
    data = smiles.from_smiles(test_smile)
    smiles = smiles.to_smiles(data)
    assert smiles == test_smile
