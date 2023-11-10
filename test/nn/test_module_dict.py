import torch

from torch_geometric.nn.module_dict import ModuleDict
from torch_geometric.testing import (
    disableExtensions,
    onlyLinux,
    withCUDA,
    withPackage,
)


def test_internal_external_key_conversion():
    assert ModuleDict.to_internal_key('a.b') == 'a#b'
    assert ModuleDict.to_internal_key('ab') == 'ab'
    assert ModuleDict.to_internal_key('a.b.c') == 'a#b#c'
    assert ModuleDict.to_internal_key(('a', 'b')) == '<a___b>'
    assert ModuleDict.to_internal_key(('a.b', 'c')) == '<a#b___c>'
    assert ModuleDict.to_internal_key('type') == '<type>'

    assert ModuleDict.to_external_key('a#b') == 'a.b'
    assert ModuleDict.to_external_key('a#b#c') == 'a.b.c'
    assert ModuleDict.to_external_key('<a___b>') == ('a', 'b')
    assert ModuleDict.to_external_key('<a#b___c>') == ('a.b', 'c')
    assert ModuleDict.to_external_key('<type>') == 'type'


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


def test_tuple_keys():
    module_dict = ModuleDict({
        ('a', 'b'): torch.nn.Linear(16, 16),
        ('a.b', 'c'): torch.nn.Linear(8, 8),
    })

    expected_keys = {('a', 'b'), ('a.b', 'c')}
    assert set(module_dict.keys()) == expected_keys
    assert set([key for key, _ in module_dict.items()]) == expected_keys

    for key in expected_keys:
        assert key in module_dict

    del module_dict['a', 'b']
    assert ('a', 'b') not in module_dict


def test_reserved_keys():
    module_dict = ModuleDict({
        'type': torch.nn.Linear(16, 16),
        '__annotations__': torch.nn.Linear(8, 8),
    })

    expected_keys = {'type', '__annotations__'}
    assert set(module_dict.keys()) == expected_keys
    assert set([key for key, _ in module_dict.items()]) == expected_keys

    for key in expected_keys:
        assert key in module_dict

    del module_dict['type']
    assert 'type' not in module_dict


@withCUDA
@onlyLinux
@disableExtensions
@withPackage('torch>=2.1.0')
def test_compile_module_dict(device):
    import torch._dynamo as dynamo

    edge_type = ("a", "to", "b")

    class TestModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.module_dict = ModuleDict({
                edge_type: torch.nn.Linear(1, 1),
            })

        def forward(self, x):
            key = ModuleDict.to_internal_key(edge_type)
            x = self.module_dict[key](x)
            return x


    x = torch.randn(1, 1, device=device)
    module = TestModule().to(device)
    explanation = dynamo.explain(module)(x)
    assert explanation.graph_break_count == 0

    # compiled_conv = torch_geometric.compile(conv)

    # expected = conv(x, edge_index)
    # out = compiled_conv(x, edge_index)
    # assert torch.allclose(out, expected)
