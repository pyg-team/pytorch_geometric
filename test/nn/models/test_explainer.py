import pytest
import torch

from torch_geometric.nn import GAT, GCN, Explainer, to_captum

try:
    from captum import attr  # noqa
    with_captum = True
except ImportError:
    with_captum = False

x = torch.randn(8, 3, requires_grad=True)
edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
                           [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6]])

GCN = GCN(3, 16, 2, 7, dropout=0.5)
GAT = GAT(3, 16, 2, 7, heads=2, concat=False)
mask_types = ['edge', 'node_and_edge', 'node']
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


@pytest.mark.parametrize('mask_type', mask_types)
@pytest.mark.parametrize('model', [GCN, GAT])
@pytest.mark.parametrize('output_idx', [None, 1])
def test_to_captum(model, mask_type, output_idx):
    captum_model = to_captum(model, mask_type=mask_type, output_idx=output_idx)
    pre_out = model(x, edge_index)
    if mask_type == 'node':
        mask = x * 0.0
        out = captum_model(mask.unsqueeze(0), edge_index)
    elif mask_type == 'edge':
        mask = torch.ones(edge_index.shape[1], dtype=torch.float,
                          requires_grad=True) * 0.5
        out = captum_model(mask.unsqueeze(0), x, edge_index)
    elif mask_type == 'node_and_edge':
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


@pytest.mark.skipif(not with_captum, reason="no 'captum' package")
@pytest.mark.parametrize('mask_type', mask_types)
@pytest.mark.parametrize('method', methods)
def test_captum_attribution_methods(mask_type, method):
    model = GCN
    captum_model = to_captum(model, mask_type, 0)
    input_mask = torch.ones((1, edge_index.shape[1]), dtype=torch.float,
                            requires_grad=True)
    explainer = getattr(attr, method)(captum_model)

    if mask_type == 'node':
        input = x.clone().unsqueeze(0)
        additional_forward_args = (edge_index, )
        sliding_window_shapes = (3, 3)
    elif mask_type == 'edge':
        input = input_mask
        additional_forward_args = (x, edge_index)
        sliding_window_shapes = (5, )
    elif mask_type == 'node_and_edge':
        input = (x.clone().unsqueeze(0), input_mask)
        additional_forward_args = (edge_index, )
        sliding_window_shapes = ((3, 3), (5, ))

    if method == 'IntegratedGradients':
        attributions, delta = explainer.attribute(
            input, target=0, internal_batch_size=1,
            additional_forward_args=additional_forward_args,
            return_convergence_delta=True)
    elif method == 'GradientShap':
        attributions, delta = explainer.attribute(
            input, target=0, return_convergence_delta=True, baselines=input,
            n_samples=1, additional_forward_args=additional_forward_args)
    elif method == 'DeepLiftShap' or method == 'DeepLift':
        attributions, delta = explainer.attribute(
            input, target=0, return_convergence_delta=True, baselines=input,
            additional_forward_args=additional_forward_args)
    elif method == 'Occlusion':
        attributions = explainer.attribute(
            input, target=0, sliding_window_shapes=sliding_window_shapes,
            additional_forward_args=additional_forward_args)
    else:
        attributions = explainer.attribute(
            input, target=0, additional_forward_args=additional_forward_args)
    if mask_type == 'node':
        assert attributions.shape == (1, 8, 3)
    elif mask_type == 'edge':
        assert attributions.shape == (1, 14)
    else:
        assert attributions[0].shape == (1, 8, 3)
        assert attributions[1].shape == (1, 14)


@pytest.mark.parametrize('model', [GCN, GAT])
def test_explainer_to_log_prob(model):
    raw_to_log = Explainer(model, return_type='raw')._to_log_prob
    prob_to_log = Explainer(model, return_type='prob')._to_log_prob
    log_to_log = Explainer(model, return_type='log_prob')._to_log_prob

    raw = torch.tensor([[1, 3.2, 6.1], [9, 9, 0.1]])
    prob = raw.softmax(dim=-1)
    log_prob = raw.log_softmax(dim=-1)

    assert torch.allclose(raw_to_log(raw), prob_to_log(prob))
    assert torch.allclose(prob_to_log(prob), log_to_log(log_prob))
