import pytest

from torch_geometric.explain.algorithm.captum import (
    CaptumHeteroModel,
    captum_output_to_dicts,
    to_captum_input,
)
from torch_geometric.nn import to_captum_model
from torch_geometric.testing import withPackage

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


@withPackage('captum')
@pytest.mark.parametrize('mask_type', mask_types)
@pytest.mark.parametrize('method', methods)
def test_captum_attribution_methods_hetero(mask_type, method, hetero_data,
                                           hetero_model):
    from captum import attr  # noqa
    data = hetero_data
    metadata = data.metadata()
    model = hetero_model(metadata)
    captum_model = to_captum_model(model, mask_type, 0, metadata)
    explainer = getattr(attr, method)(captum_model)
    assert isinstance(captum_model, CaptumHeteroModel)

    args = ['additional_arg1']
    input, additional_forward_args = to_captum_input(data.x_dict,
                                                     data.edge_index_dict,
                                                     mask_type, *args)
    if mask_type == 'node':
        sliding_window_shapes = ((3, 3), (3, 3))
    elif mask_type == 'edge':
        sliding_window_shapes = ((5, ), (5, ), (5, ))
    else:
        sliding_window_shapes = ((3, 3), (3, 3), (5, ), (5, ), (5, ))

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
        assert len(attributions) == len(metadata[0])
        x_attr_dict, _ = captum_output_to_dicts(attributions, mask_type,
                                                metadata)
        for node_type in metadata[0]:
            num_nodes = data[node_type].num_nodes
            num_node_feats = data[node_type].x.shape[1]
            assert x_attr_dict[node_type].shape == (num_nodes, num_node_feats)
    elif mask_type == 'edge':
        assert len(attributions) == len(metadata[1])
        _, edge_attr_dict = captum_output_to_dicts(attributions, mask_type,
                                                   metadata)
        for edge_type in metadata[1]:
            num_edges = data[edge_type].edge_index.shape[1]
            assert edge_attr_dict[edge_type].shape == (num_edges, )
    else:
        assert len(attributions) == len(metadata[0]) + len(metadata[1])
        x_attr_dict, edge_attr_dict = captum_output_to_dicts(
            attributions, mask_type, metadata)
        for edge_type in metadata[1]:
            num_edges = data[edge_type].edge_index.shape[1]
            assert edge_attr_dict[edge_type].shape == (num_edges, )

        for node_type in metadata[0]:
            num_nodes = data[node_type].num_nodes
            num_node_feats = data[node_type].x.shape[1]
            assert x_attr_dict[node_type].shape == (num_nodes, num_node_feats)
