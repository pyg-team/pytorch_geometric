import torch

from torch_geometric.nn.module_dict import ModuleDict


def test_internal_external_key_conversion():
    assert ModuleDict.to_internal_key('a.b') == 'a#b'
    assert ModuleDict.to_internal_key('ab') == 'ab'
    assert ModuleDict.to_internal_key('a.b.c') == 'a#b#c'

    assert ModuleDict.to_external_key('a#b') == 'a.b'
    assert ModuleDict.to_external_key('a#b#c') == 'a.b.c'


def test_dot_syntax_keys():
    module_dict = ModuleDict({
        'lin1': torch.nn.Linear(16, 16),
        'model.lin2': torch.nn.Linear(8, 8),
        'model.sub_model.lin3': torch.nn.Linear(4, 4),
    })

    expected_keys = {'lin1', 'model.lin2', 'model.sub_model.lin3'}
    assert set(module_dict.keys()) == expected_keys
    assert set([key for key, _ in module_dict.items()]) == expected_keys

    for key in expected_keys:
        assert key in module_dict

    del module_dict['model.lin2']
    assert 'model.lin2' not in module_dict
