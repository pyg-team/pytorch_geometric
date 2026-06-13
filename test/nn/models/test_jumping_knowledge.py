import torch

from torch_geometric.nn import HeteroJumpingKnowledge, JumpingKnowledge


def test_jumping_knowledge():
    num_nodes, channels, num_layers = 100, 17, 5
    xs = list([torch.randn(num_nodes, channels) for _ in range(num_layers)])

    model = JumpingKnowledge('cat')
    assert str(model) == 'JumpingKnowledge(cat)'

    out = model(xs)
    assert out.size() == (num_nodes, channels * num_layers)

    model = JumpingKnowledge('max')
    assert str(model) == 'JumpingKnowledge(max)'

    out = model(xs)
    assert out.size() == (num_nodes, channels)

    model = JumpingKnowledge('lstm', channels, num_layers)
    assert str(model) == (f'JumpingKnowledge(lstm, channels='
                          f'{channels}, layers={num_layers})')

    out = model(xs)
    assert out.size() == (num_nodes, channels)


def test_hetero_jumping_knowledge():
    num_nodes, channels, num_layers = 100, 17, 5

    types = ["author", "paper"]
    xs_dict = {
        key: [torch.randn(num_nodes, channels) for _ in range(num_layers)]
        for key in types
    }

    model = HeteroJumpingKnowledge(types, mode='cat')
    model.reset_parameters()
    assert str(model) == 'HeteroJumpingKnowledge(num_types=2, mode=cat)'

    out_dict = model(xs_dict)
    for out in out_dict.values():
        assert out.size() == (num_nodes, channels * num_layers)

    model = HeteroJumpingKnowledge(types, mode='max')
    assert str(model) == 'HeteroJumpingKnowledge(num_types=2, mode=max)'

    out_dict = model(xs_dict)
    for out in out_dict.values():
        assert out.size() == (num_nodes, channels)

    model = HeteroJumpingKnowledge(types, mode='lstm', channels=channels,
                                   num_layers=num_layers)
    assert str(model) == (f'HeteroJumpingKnowledge(num_types=2, mode=lstm, '
                          f'channels={channels}, layers={num_layers})')

    out_dict = model(xs_dict)
    for out in out_dict.values():
        assert out.size() == (num_nodes, channels)
