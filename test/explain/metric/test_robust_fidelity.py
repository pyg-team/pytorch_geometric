import pytest
import torch

from torch_geometric.explain import (
    DummyExplainer,
    Explainer,
    robust_fidelity,
)


class DummyNodeModel(torch.nn.Module):
    def forward(self, x, edge_index):
        return x


class DummyGraphModel(torch.nn.Module):
    def forward(self, x, edge_index):
        return x[0, :].view(1, -1)


@pytest.mark.parametrize('explanation_type', ['model'])
def test_robust_fidelity(explanation_type):
    x = torch.randn(8, 4)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
        [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6],
    ])

    explainer = Explainer(
        DummyNodeModel(),
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

    target = torch.randint(0, x.size(1), (x.size(0),))

    explanation = explainer(x, edge_index, index=0,
                            target=target)

    pos_fid, neg_fid, del_fid, pos_fid_l, neg_fid_l, del_fid_l = \
        robust_fidelity(explainer, explanation, undirect=False)

    x = torch.randn(8, 4)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
        [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6],
    ])

    explainer = Explainer(
        DummyGraphModel(),
        algorithm=DummyExplainer(),
        explanation_type=explanation_type,
        node_mask_type='object',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            return_type='raw',
            task_level='graph',
        ),
    )

    target = torch.randint(0, x.size(1), (1,))

    explanation = explainer(x, edge_index, target=target)

    pos_fid, neg_fid, del_fid, pos_fid_l, neg_fid_l, del_fid_l = \
        robust_fidelity(explainer, explanation, undirect=True)
