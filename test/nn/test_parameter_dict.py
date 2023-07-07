import torch

from torch_geometric.nn.parameter_dict import ParameterDict


def test_internal_external_key_conversion():
    assert ParameterDict.to_internal_key('a.b') == 'a#b'
    assert ParameterDict.to_internal_key('ab') == 'ab'
    assert ParameterDict.to_internal_key('a.b.c') == 'a#b#c'
    assert ParameterDict.to_internal_key(('a', 'b')) == '<a___b>'
    assert ParameterDict.to_internal_key(('a.b', 'c')) == '<a#b___c>'

    assert ParameterDict.to_external_key('a#b') == 'a.b'
    assert ParameterDict.to_external_key('a#b#c') == 'a.b.c'
    assert ParameterDict.to_external_key('<a___b>') == ('a', 'b')
    assert ParameterDict.to_external_key('<a#b___c>') == ('a.b', 'c')


def test_dot_syntax_keys():
    parameter_dict = {
        'param1': torch.nn.Parameter(torch.randn(16, 16)),
        'model.param2': torch.nn.Parameter(torch.randn(8, 8)),
        'model.sub_model.param3': torch.nn.Parameter(torch.randn(4, 4)),
    }
    parameter_dict = ParameterDict(parameter_dict)

    expected_keys = {'param1', 'model.param2', 'model.sub_model.param3'}
    assert set(parameter_dict.keys()) == expected_keys
    assert set([key for key, _ in parameter_dict.items()]) == expected_keys

    for key in expected_keys:
        assert key in parameter_dict

    assert 'model.param2' in parameter_dict
    del parameter_dict['model.param2']
    assert 'model.param2' not in parameter_dict


def test_tuple_keys():
    parameter_dict = {
        ('a', 'b'): torch.nn.Parameter(torch.randn(16, 16)),
        ('a.b', 'c'): torch.nn.Parameter(torch.randn(8, 8)),
    }
    parameter_dict = ParameterDict(parameter_dict)

    expected_keys = {('a', 'b'), ('a.b', 'c')}
    assert set(parameter_dict.keys()) == expected_keys
    assert set([key for key, _ in parameter_dict.items()]) == expected_keys

    for key in expected_keys:
        assert key in parameter_dict

    assert ('a', 'b') in parameter_dict
    del parameter_dict['a', 'b']
    assert ('a', 'b') not in parameter_dict
