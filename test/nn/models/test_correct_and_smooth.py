import torch
from torch_sparse import SparseTensor

from torch_geometric.nn.models import CorrectAndSmooth


def test_correct_and_smooth():
    y_soft = torch.tensor([0.1, 0.5, 0.4]).repeat(6, 1)
    y_true = torch.tensor([1, 0, 0, 2, 1, 1])
    edge_index = torch.tensor([[0, 1, 1, 2, 4, 5], [1, 0, 2, 1, 5, 4]])
    adj = SparseTensor.from_edge_index(edge_index, sparse_sizes=(6, 6)).t()
    mask = torch.randint(0, 2, (6, ), dtype=torch.bool)

    model = CorrectAndSmooth(
        num_correction_layers=2,
        correction_alpha=0.5,
        num_smoothing_layers=2,
        smoothing_alpha=0.5,
    )
    assert str(model) == ('CorrectAndSmooth(\n'
                          '  correct: num_layers=2, alpha=0.5\n'
                          '  smooth:  num_layers=2, alpha=0.5\n'
                          '  autoscale=True, scale=1.0\n'
                          ')')

    out = model.correct(y_soft, y_true[mask], mask, edge_index)
    assert out.size() == (6, 3)
    assert torch.allclose(out, model.correct(y_soft, y_true[mask], mask, adj))

    out = model.smooth(y_soft, y_true[mask], mask, edge_index)
    assert out.size() == (6, 3)
    assert torch.allclose(out, model.smooth(y_soft, y_true[mask], mask, adj))

    # Test without autoscale:
    model = CorrectAndSmooth(
        num_correction_layers=2,
        correction_alpha=0.5,
        num_smoothing_layers=2,
        smoothing_alpha=0.5,
        autoscale=False,
    )
    out = model.correct(y_soft, y_true[mask], mask, edge_index)
    assert out.size() == (6, 3)
    assert torch.allclose(out, model.correct(y_soft, y_true[mask], mask, adj))
