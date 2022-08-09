import torch

from torch_geometric.utils import scatter_composite


def test_std():
    src = torch.tensor([[2, 0, 1, 4, 3], [0, 2, 1, 3, 4]], dtype=torch.float)
    src.requires_grad_()
    index = torch.tensor([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.long)

    out = scatter_composite(src, index, dim=-1, unbiased=True, reduce='std')
    std = src.std(dim=-1, unbiased=True)[0]
    expected = torch.tensor([[std, 0], [0, std]])
    assert torch.allclose(out, expected)

    out.backward(torch.randn_like(out))

    jit = torch.jit.script(scatter_composite)
    assert jit(src, index, dim=-1, unbiased=True,
               reduce='std').tolist() == out.tolist()
