import pytest
import torch

from torch_geometric.data import HeteroData
from torch_geometric.nn import (
    GCNConv,
    HeteroConv,
    SAGEConv,
    captum_output_to_dicts,
    to_captum_input,
    to_captum_model,
    to_hetero,
)
from torch_geometric.nn.models.explainer import (
    CaptumHeteroModel,
    clear_masks,
    set_hetero_masks,
)
from torch_geometric.testing import withPackage
from torch_geometric.typing import Metadata

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


def get_edge_index(num_src_nodes, num_dst_nodes, num_edges):
    row = torch.randint(num_src_nodes, (num_edges, ), dtype=torch.long)
    col = torch.randint(num_dst_nodes, (num_edges, ), dtype=torch.long)
    return torch.stack([row, col], dim=0)


def get_hetero_data():
    data = HeteroData()
    data['paper'].x = torch.randn(8, 16)
    data['author'].x = torch.randn(10, 8)
    data['paper', 'paper'].edge_index = get_edge_index(8, 8, 10)
    data['author', 'paper'].edge_index = get_edge_index(10, 8, 10)
    data['paper', 'author'].edge_index = get_edge_index(8, 10, 10)
    return data


class HeteroModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = HeteroConv({
            ('paper', 'to', 'paper'):
            GCNConv(-1, 32),
            ('author', 'to', 'paper'):
            SAGEConv((-1, -1), 32),
            ('paper', 'to', 'author'):
            SAGEConv((-1, -1), 32),
        })

        self.conv2 = HeteroConv({
            ('paper', 'to', 'paper'):
            GCNConv(-1, 32),
            ('author', 'to', 'paper'):
            SAGEConv((-1, -1), 32),
            ('paper', 'to', 'author'):
            SAGEConv((-1, -1), 32),
        })


class GraphSAGE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), 32)
        self.conv2 = SAGEConv((-1, -1), 32)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


class HeteroSAGE(torch.nn.Module):
    def __init__(self, metadata: Metadata):
        super().__init__()
        self.graph_sage = to_hetero(GraphSAGE(), metadata, debug=False)

    def forward(self, x_dict, edge_index_dict,
                additonal_arg=None) -> torch.Tensor:
        # Make sure additonal args gets passed.
        assert additonal_arg is not None
        return self.graph_sage(x_dict, edge_index_dict)['paper']


def test_set_clear_mask():
    data = get_hetero_data()
    edge_mask_dict = {
        ('paper', 'to', 'paper'): torch.ones(200),
        ('author', 'to', 'paper'): torch.ones(100),
        ('paper', 'to', 'author'): torch.ones(100),
    }

    model = HeteroModel()

    set_hetero_masks(model, edge_mask_dict, data.edge_index_dict)
    for edge_type in data.edge_types:  # Check that masks are correctly set:
        str_edge_type = '__'.join(edge_type)
        assert torch.allclose(model.conv1.convs[str_edge_type]._edge_mask,
                              edge_mask_dict[edge_type])
        assert model.conv1.convs[str_edge_type].explain

    clear_masks(model)
    for edge_type in data.edge_types:
        str_edge_type = '__'.join(edge_type)
        assert model.conv1.convs[str_edge_type]._edge_mask is None
        assert not model.conv1.convs[str_edge_type].explain

    model = GraphSAGE()
    model = to_hetero(GraphSAGE(), data.metadata(), debug=False)

    set_hetero_masks(model, edge_mask_dict, data.edge_index_dict)
    for edge_type in data.edge_types:  # Check that masks are correctly set:
        str_edge_type = '__'.join(edge_type)
        assert torch.allclose(model.conv1[str_edge_type]._edge_mask,
                              edge_mask_dict[edge_type])
        assert model.conv1[str_edge_type].explain

    clear_masks(model)
    for edge_type in data.edge_types:
        str_edge_type = '__'.join(edge_type)
        assert model.conv1[str_edge_type]._edge_mask is None
        assert not model.conv1[str_edge_type].explain


@withPackage('captum')
@pytest.mark.parametrize('mask_type', mask_types)
@pytest.mark.parametrize('method', methods)
def test_captum_attribution_methods_hetero(mask_type, method):
    from captum import attr  # noqa
    data = get_hetero_data()
    metadata = data.metadata()
    model = HeteroSAGE(metadata)
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
