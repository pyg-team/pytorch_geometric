import torch
from torch_geometric.nn import DeepGCNs


def test_deepgcns():
    model = DeepGCNs('res+')
    assert model.__repr__() == 'DeepGCNs(res+)'

    model = DeepGCNs('res')
    assert model.__repr__() == 'DeepGCNs(res)'

    model = DeepGCNs('dense')
    assert model.__repr__() == 'DeepGCNs(dense)'

    model = DeepGCNs('plain')
    assert model.__repr__() == 'DeepGCNs(plain)'
