import torch

from torch_geometric.utils import scatter_composite


def test_logsumexp():
    inputs = torch.tensor([
        0.5, 0.5, 0.0, -2.1, 3.2, 7.0, -1.0, -100.0,
        float('-inf'),
        float('-inf'), 0.0
    ])
    inputs.requires_grad_()
    index = torch.tensor([0, 0, 1, 1, 1, 2, 4, 4, 5, 6, 6])
    splits = [2, 3, 1, 0, 2, 1, 2]

    outputs = scatter_composite(inputs, index, reduce='logsumexp')

    for src, out in zip(inputs.split(splits), outputs.unbind()):
        assert out.tolist() == torch.logsumexp(src, dim=0).tolist()

    outputs.backward(torch.randn_like(outputs))

    jit = torch.jit.script(scatter_composite)
    assert jit(inputs, index, reduce='logsumexp').tolist() == outputs.tolist()
