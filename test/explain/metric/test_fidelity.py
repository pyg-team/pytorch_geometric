import pytest
import torch

from torch_geometric.explain import (
    DummyExplainer,
    Explainer,
    characterization_score,
    fidelity,
    fidelity_curve_auc,
)


class DummyModel(torch.nn.Module):
    def forward(self, x, edge_index):
        return x


@pytest.mark.parametrize('explanation_type', ['model', 'phenomenon'])
def test_fidelity(explanation_type):
    x = torch.randn(8, 4)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
        [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6],
    ])

    explainer = Explainer(
        DummyModel(),
        algorithm=DummyExplainer(),
        explanation_type=explanation_type,
        node_mask_type='object',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            return_type='raw',
            task_level='node',
        ),
    )

    target = None
    if explanation_type == 'phenomenon':
        target = torch.randint(0, x.size(1), (x.size(0), ))

    explanation = explainer(x, edge_index, target=target,
                            index=torch.arange(4))

    pos_fidelity, neg_fidelity = fidelity(explainer, explanation)
    assert pos_fidelity == 0.0 and neg_fidelity == 0.0


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
