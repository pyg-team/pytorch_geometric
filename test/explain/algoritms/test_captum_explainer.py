import pytest
import torch

from torch_geometric.explain.algorithm import CaptumAlgorithm
from torch_geometric.explain.config import (
    ExplainerConfig,
    MaskType,
    ModelConfig,
)
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

    explainer_config = ExplainerConfig(explanation_type="model",
                                       node_mask_type=node_mask_type,
                                       edge_mask_type=edge_mask_type)
    model_config = ModelConfig(mode="classification", return_type="raw",
                               task_level="graph")
    target = GCN(x, edge_index).max(dim=-1)
    explanation = explainer(model=GCN, x=x, edge_index=edge_index,
                            target_index=0, explainer_config=explainer_config,
                            model_config=model_config, target=target)

    if node_mask_type is not None:
        assert explanation.node_features_mask.shape == (8, 3)
    if edge_mask_type is not None:
        assert explanation.edge_mask.shape == (1, 14)
