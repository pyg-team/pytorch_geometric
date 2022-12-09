import torch

from torch_geometric.explain.utils.metrics import (
    get_characterization_score,
    get_fidelity_curve_auc,
)


def test_characterization_score():
    weights_1 = torch.tensor([0.2, 0.8])
    weights_2 = torch.tensor([1, 4])
    components = torch.tensor([[1, 0.6, 0.5, 1], [0, 0.2, 0.5, 1]])

    assert get_characterization_score(
        components, weights_1).tolist() == [1.0, 0.75, 0.5, 0.0]
    assert get_characterization_score(
        components, weights_2).tolist() == [1.0, 0.75, 0.5, 0.0]


def test_fidelity_curve_auc():

    fidelities = torch.tensor([[1, 1, 0.5, 1], [0, 0.5, 0.5, 0.9]])
    x_1 = torch.tensor([0, 1, 2, 3])
    x_2 = torch.tensor([10, 11, 12, 13])
    x_3 = torch.tensor([0, 1, 2, 5])

    assert round(get_fidelity_curve_auc(fidelities, x_1).item(), 4) == 8.5
    assert round(get_fidelity_curve_auc(fidelities, x_2).item(), 4) == 8.5
    assert round(get_fidelity_curve_auc(fidelities, x_3).item(), 4) == 19.5
