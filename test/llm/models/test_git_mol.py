import torch

from torch_geometric.llm.models import GITMol
from torch_geometric.testing import withPackage


@withPackage('transformers', 'sentencepiece', 'accelerate')
def test_git_mol():
    model = GITMol()

    x = torch.ones(10, 16, dtype=torch.long)
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    ])
    edge_attr = torch.zeros(edge_index.size(1), 16, dtype=torch.long)
    batch = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    smiles = ['CC(C)([C@H]1CC2=C(O1)C=CC3=C2OC(=O)C=C3)O']
    captions = ['The molecule is the (R)-(-)-enantiomer of columbianetin.']
    images = torch.randn(1, 3, 224, 224)

    # Test train:
    loss = model(x, edge_index, batch, edge_attr, smiles, images, captions)
    assert loss >= 0
