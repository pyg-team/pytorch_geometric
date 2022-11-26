import pytest
import torch

from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import GAT, GCN, Explainer, SAGEConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.models import to_captum_input, to_captum_model
from torch_geometric.testing import withPackage

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
    captum_model = to_captum_model(model, mask_type=mask_type,
                                   output_idx=output_idx)
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


@withPackage('captum')
@pytest.mark.parametrize('mask_type', mask_types)
@pytest.mark.parametrize('method', methods)
def test_captum_attribution_methods(mask_type, method):
    from captum import attr  # noqa

    captum_model = to_captum_model(GCN, mask_type, 0)
    explainer = getattr(attr, method)(captum_model)
    data = Data(x, edge_index)
    input, additional_forward_args = to_captum_input(data.x, data.edge_index,
                                                     mask_type)
    if mask_type == 'node':
        sliding_window_shapes = (3, 3)
    elif mask_type == 'edge':
        sliding_window_shapes = (5, )
    elif mask_type == 'node_and_edge':
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
        assert attributions[0].shape == (1, 8, 3)
    elif mask_type == 'edge':
        assert attributions[0].shape == (1, 14)
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


def test_custom_explain_message():
    x = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])

    conv = SAGEConv(8, 32)

    def explain_message(self, inputs, x_i, x_j):
        assert isinstance(self, SAGEConv)
        assert inputs.size() == (6, 8)
        assert inputs.size() == x_i.size() == x_j.size()
        assert torch.allclose(inputs, x_j)
        self.x_i = x_i
        self.x_j = x_j
        return inputs

    conv.explain_message = explain_message.__get__(conv, MessagePassing)
    conv.explain = True

    conv(x, edge_index)

    assert torch.allclose(conv.x_i, x[edge_index[1]])
    assert torch.allclose(conv.x_j, x[edge_index[0]])


@withPackage('captum')
@pytest.mark.parametrize('mask_type', ['node', 'edge', 'node_and_edge'])
def test_to_captum_input(mask_type):
    num_nodes = x.shape[0]
    num_node_feats = x.shape[1]
    num_edges = edge_index.shape[1]

    # Check for Data:
    data = Data(x, edge_index)
    args = 'test_args'
    inputs, additional_forward_args = to_captum_input(data.x, data.edge_index,
                                                      mask_type, args)
    if mask_type == 'node':
        assert len(inputs) == 1
        assert inputs[0].shape == (1, num_nodes, num_node_feats)
        assert len(additional_forward_args) == 2
        assert torch.allclose(additional_forward_args[0], edge_index)
    elif mask_type == 'edge':
        assert len(inputs) == 1
        assert inputs[0].shape == (1, num_edges)
        assert inputs[0].sum() == num_edges
        assert len(additional_forward_args) == 3
        assert torch.allclose(additional_forward_args[0], x)
        assert torch.allclose(additional_forward_args[1], edge_index)
    else:
        assert len(inputs) == 2
        assert inputs[0].shape == (1, num_nodes, num_node_feats)
        assert inputs[1].shape == (1, num_edges)
        assert inputs[1].sum() == num_edges
        assert len(additional_forward_args) == 2
        assert torch.allclose(additional_forward_args[0], edge_index)

    # Check for HeteroData:
    data = HeteroData()
    x2 = torch.rand(8, 3)
    data['paper'].x = x
    data['author'].x = x2
    data['paper', 'to', 'author'].edge_index = edge_index
    data['author', 'to', 'paper'].edge_index = edge_index.flip([0])
    inputs, additional_forward_args = to_captum_input(data.x_dict,
                                                      data.edge_index_dict,
                                                      mask_type, args)
    if mask_type == 'node':
        assert len(inputs) == 2
        assert inputs[0].shape == (1, num_nodes, num_node_feats)
        assert inputs[1].shape == (1, num_nodes, num_node_feats)
        assert len(additional_forward_args) == 2
        for key in data.edge_types:
            torch.allclose(additional_forward_args[0][key],
                           data[key].edge_index)
    elif mask_type == 'edge':
        assert len(inputs) == 2
        assert inputs[0].shape == (1, num_edges)
        assert inputs[1].shape == (1, num_edges)
        assert inputs[1].sum() == inputs[0].sum() == num_edges
        assert len(additional_forward_args) == 3
        for key in data.node_types:
            torch.allclose(additional_forward_args[0][key], data[key].x)
        for key in data.edge_types:
            torch.allclose(additional_forward_args[1][key],
                           data[key].edge_index)
    else:
        assert len(inputs) == 4
        assert inputs[0].shape == (1, num_nodes, num_node_feats)
        assert inputs[1].shape == (1, num_nodes, num_node_feats)
        assert inputs[2].shape == (1, num_edges)
        assert inputs[3].shape == (1, num_edges)
        assert inputs[3].sum() == inputs[2].sum() == num_edges
        assert len(additional_forward_args) == 2
        for key in data.edge_types:
            torch.allclose(additional_forward_args[0][key],
                           data[key].edge_index)
