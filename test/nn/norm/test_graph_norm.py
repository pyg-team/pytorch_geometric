import torch

from torch_geometric.nn import GraphNorm
from torch_geometric.testing import is_full_test, withDevice


@withDevice
def test_graph_norm(device):
    torch.manual_seed(42)
    x = torch.randn(200, 16, device=device)
    batch = torch.arange(4, device=device).view(-1, 1).repeat(1, 50).view(-1)

    norm = GraphNorm(16, device=device)
    assert str(norm) == 'GraphNorm(16)'

    if is_full_test():
        torch.jit.script(norm)

    out = norm(x)
    assert out.size() == (200, 16)
    assert torch.allclose(out.mean(dim=0), torch.zeros(16, device=device),
                          atol=1e-6)
    assert torch.allclose(out.std(dim=0, unbiased=False),
                          torch.ones(16, device=device), atol=1e-6)

    out = norm(x, batch)
    assert out.size() == (200, 16)
    assert torch.allclose(out[:50].mean(dim=0), torch.zeros(16, device=device),
                          atol=1e-6)
    assert torch.allclose(out[:50].std(dim=0, unbiased=False),
                          torch.ones(16, device=device), atol=1e-6)
