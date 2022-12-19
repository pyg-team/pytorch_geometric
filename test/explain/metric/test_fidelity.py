import torch

from torch_geometric.explain import characterization_score, fidelity_curve_auc


def test_characterization_score():
    out = characterization_score(
        pos_fidelity=torch.tensor([1.0, 0.6, 0.5, 1.0]),
        neg_fidelity=torch.tensor([0.0, 0.2, 0.5, 1.0]),
        pos_weight=0.2,
        neg_weight=0.8,
    )
    assert out.tolist() == [1.0, 0.75, 0.5, 0.0]


def test_fidelity_curve_auc():
    pos_fidelity = torch.tensor([1.0, 1.0, 0.5, 1.0])
    neg_fidelity = torch.tensor([0.0, 0.5, 0.5, 0.9])

    x = torch.tensor([0, 1, 2, 3])
    out = round(float(fidelity_curve_auc(pos_fidelity, neg_fidelity, x)), 4)
    assert out == 8.5

    x = torch.tensor([10, 11, 12, 13])
    out = round(float(fidelity_curve_auc(pos_fidelity, neg_fidelity, x)), 4)
    assert out == 8.5

    x = torch.tensor([0, 1, 2, 5])
    out = round(float(fidelity_curve_auc(pos_fidelity, neg_fidelity, x)), 4)
    assert out == 19.5
