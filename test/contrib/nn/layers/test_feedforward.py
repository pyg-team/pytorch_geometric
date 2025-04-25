# isort: skip_file
# yapf: disable
import torch

from torch_geometric.contrib.nn.layers.feedforward import (
    PositionwiseFeedForward,
)


def test_positionwise_feedforward():
    hidden_dim = 8
    ffn_hidden_dim = 32
    dropout = 0
    model = PositionwiseFeedForward(hidden_dim, ffn_hidden_dim,
                                    dropout=dropout)
    x = torch.randn(4, 10, hidden_dim)
    out = model(x)
    assert out.shape == x.shape, "Output shape must match input shape."
    direct = model.fc2(model.fc1(x))
    assert not torch.allclose(
        out, direct), "Output should differ from direct transformation."

# yapf: enable
