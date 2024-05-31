import torch

from torch_geometric.nn import HeteroJK, JumpingKnowledge
from torch_geometric.testing import is_full_test


def test_jumping_knowledge():
    num_nodes, channels, num_layers = 100, 17, 5
    xs = list([torch.randn(num_nodes, channels) for _ in range(num_layers)])

    model = JumpingKnowledge('cat')
    assert str(model) == 'JumpingKnowledge(cat)'

    out = model(xs)
    assert out.size() == (num_nodes, channels * num_layers)

    if is_full_test():
        jit = torch.jit.script(model)
        assert torch.allclose(jit(xs), out)

    model = JumpingKnowledge('max')
    assert str(model) == 'JumpingKnowledge(max)'

    out = model(xs)
    assert out.size() == (num_nodes, channels)

    if is_full_test():
        jit = torch.jit.script(model)
        assert torch.allclose(jit(xs), out)

    model = JumpingKnowledge('lstm', channels, num_layers)
    assert str(model) == (f'JumpingKnowledge(lstm, channels='
                          f'{channels}, layers={num_layers})')

    out = model(xs)
    assert out.size() == (num_nodes, channels)

    if is_full_test():
        jit = torch.jit.script(model)
        assert torch.allclose(jit(xs), out)


def test_hetero_jk():
    num_nodes, channels, num_layers = 100, 17, 5

    node_types = ["author", "paper"]
    num_node_types = len(node_types)

    xs_dict = {}
    for node_t in node_types:
        xs_dict[node_t] = [
            torch.randn(num_nodes, channels) for _ in range(num_layers)
        ]

    # Cat mode assertions
    model = HeteroJK(node_types, 'cat')
    assert str(model) == f'HeteroJK({num_node_types=}, mode=cat)'

    out_dict = model(xs_dict)
    for out in out_dict.values():
        assert out.size() == (num_nodes, channels * num_layers)

    if is_full_test():
        jit = torch.jit.script(model)
        jit_out = jit(xs_dict)
        for node_t in node_types:
            assert torch.allclose(jit_out[node_t], out_dict[node_t])

    # Max mode assertions
    model = HeteroJK(node_types, 'max')
    assert str(model) == f'HeteroJK({num_node_types=}, mode=max)'

    out_dict = model(xs_dict)
    for out in out_dict.values():
        assert out.size() == (num_nodes, channels)

    if is_full_test():
        jit = torch.jit.script(model)
        jit_out = jit(xs_dict)
        for node_t in node_types:
            assert torch.allclose(jit_out[node_t], out_dict[node_t])

    # LSTM mode assertions
    model = HeteroJK(node_types, 'lstm', channels=channels,
                     num_layers=num_layers)
    assert str(model) == (f'HeteroJK({num_node_types=}, mode=lstm, channels='
                          f'{channels}, layers={num_layers})')

    out_dict = model(xs_dict)
    for out in out_dict.values():
        assert out.size() == (num_nodes, channels)

    if is_full_test():
        jit = torch.jit.script(model)
        jit_out = jit(xs_dict)
        for node_t in node_types:
            assert torch.allclose(jit_out[node_t], out_dict[node_t])
