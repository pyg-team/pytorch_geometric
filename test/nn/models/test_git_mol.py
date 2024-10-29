import torch

from torch_geometric.nn.models import GITMol


def test_git_mol():
    model = GITMol()

    x = torch.randn(10, 16)
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
    ])
    edge_attr = torch.randn(edge_index.size(1), 16)
    batch = torch.zeros(x.size(0), dtype=torch.long)
    smiles = ['CC(C)([C@H]1CC2=C(O1)C=CC3=C2OC(=O)C=C3)O']
    captions = ['The molecule is the (R)-(-)-enantiomer of columbianetin.']
    images = torch.randn(1, 3, 224, 224)

    # Test train:
    loss = model(x, edge_index, batch, edge_attr, smiles, captions, images)
    assert loss >= 0

    # Test inference:
    pred = model.inference(x, edge_index, batch, edge_attr, smiles, captions)
    assert len(pred) == 1
