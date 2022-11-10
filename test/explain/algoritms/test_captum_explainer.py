import pytest
import torch

from torch_geometric.explain.algorithm.captum_explainer import (
    CaptumAlgorithm,
    to_captum,
)
from torch_geometric.explain.config import MaskType
from torch_geometric.nn import GAT, GCN
from torch_geometric.testing import withPackage

x = torch.randn(8, 3, requires_grad=True)
edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
                           [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6]])

GCN = GCN(3, 16, 2, 7, dropout=0.5)
GAT = GAT(3, 16, 2, 7, heads=2, concat=False)
node_mask_types = [MaskType.attributes]
edge_mask_types = [None, MaskType.object]
methods = [
    'Saliency',
    'InputXGradient',
    'Deconvolution',
    'FeatureAblation',
    'ShapleyValueSampling',
    'IntegratedGradients',
    'GradientShap',
    'Occlusion',
    'GuidedBackprop',
    'KernelShap',
    'Lime',
]


@pytest.mark.parametrize('node_mask_type', node_mask_types)
@pytest.mark.parametrize('edge_mask_type', edge_mask_types)
@pytest.mark.parametrize('model', [GCN, GAT])
@pytest.mark.parametrize('output_idx', [None, 1])
def test_to_captum(model, node_mask_type, edge_mask_type, output_idx):
    captum_model = to_captum(model, node_mask_type=node_mask_type,
                             edge_mask_type=edge_mask_type,
                             output_idx=output_idx)
    pre_out = model(x, edge_index)
    if edge_mask_type is None:
        mask = x * 0.0
        out = captum_model(mask.unsqueeze(0), edge_index)
    elif edge_mask_type == MaskType.object and node_mask_type is None:
        mask = torch.ones(edge_index.shape[1], dtype=torch.float,
                          requires_grad=True) * 0.5
        out = captum_model(mask.unsqueeze(0), x, edge_index)
    else:
        node_mask = x * 0.0
        edge_mask = torch.ones(edge_index.shape[1], dtype=torch.float,
                               requires_grad=True) * 0.5
        out = captum_model(node_mask.unsqueeze(0), edge_mask.unsqueeze(0),
                           edge_index)

    if output_idx is not None:
        assert out.shape == (1, 7)
        assert torch.any(out != pre_out[[output_idx]])
    else:
        assert out.shape == (8, 7)
        assert torch.any(out != pre_out)


@withPackage('captum')
@pytest.mark.parametrize('node_mask_type', node_mask_types)
@pytest.mark.parametrize('edge_mask_type', edge_mask_types)
@pytest.mark.parametrize('method', methods)
def test_captum_attribution_methods(node_mask_type, edge_mask_type, method):

    if method == "Occlusion":
        if node_mask_type is None:
            sliding_window_shapes = (5, )
        elif edge_mask_type is None:
            sliding_window_shapes = (3, 3)
        else:
            sliding_window_shapes = ((3, 3), (5, ))
        explainer = CaptumAlgorithm(
            method, sliding_window_shapes=sliding_window_shapes)
    else:
        explainer = CaptumAlgorithm(method)

    explanation = explainer(model=GCN, x=x, edge_index=edge_index,
                            target_index=0, node_mask_type=node_mask_type,
                            edge_mask_type=edge_mask_type)

    if node_mask_type is not None:
        assert explanation.node_features_mask.shape == (8, 3)
    if edge_mask_type is not None:
        assert explanation.edge_mask.shape == (1, 14)


def test_integration_captum_alg_explainer():
    explainer = CaptumAlgorithm('IntegratedGradients')
    explanation = explainer(model=GCN, x=x, edge_index=edge_index,
                            target_index=0, node_mask_type=None,
                            edge_mask_type=MaskType.object)

    assert explanation.edge_mask.shape == (1, 14)
