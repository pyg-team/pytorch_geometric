import torch
from torch_geometric.nn import LayerNorm

def test_layer_norm():
    ch = 16
    x = torch.randn(100, ch)
    batch = torch.zeros(100, dtype=torch.long)

    norm = LayerNorm(in_channels=ch, affine=False)
    assert norm.__repr__() == 'LayerNorm()'
    norm = torch.jit.script(norm)
    out1 = norm(x)
    assert out1.size() == (100, ch)

    out2 = norm(torch.cat([x, x], dim=0), torch.cat([batch, batch + 1], dim=0))
    assert torch.allclose(out1, out2[:100])
    assert torch.allclose(out1, out2[100:])

    norm = LayerNorm(in_channels=ch, affine=True)
    assert norm.__repr__() == 'LayerNorm()'
    norm = torch.jit.script(norm)
    out1 = norm(x)
    assert out1.size() == (100, ch)

    out2 = norm(torch.cat([x, x], dim=0), torch.cat([batch, batch + 1], dim=0))
    assert torch.allclose(out1, out2[:100])
    assert torch.allclose(out1, out2[100:])