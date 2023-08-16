from typing import Optional

import pytest
import torch

from torch_geometric.explain import Explainer, Explanation
from torch_geometric.explain.algorithm import CaptumExplainer
from torch_geometric.explain.config import (
    MaskType,
    ModelConfig,
    ModelMode,
    ModelReturnType,
    ModelTaskLevel,
)
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.testing import withPackage

methods = [
    'Saliency',
    'InputXGradient',
    'Deconvolution',
    'ShapleyValueSampling',
    'IntegratedGradients',
    'GuidedBackprop',
]

unsupported_methods = [
    'FeatureAblation',
    'Occlusion',
    'DeepLift',
    'DeepLiftShap',
    'GradientShap',
    'KernelShap',
    'Lime',
]


class GCN(torch.nn.Module):
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.model_config = model_config

        if model_config.mode == ModelMode.multiclass_classification:
            out_channels = 7
        else:
            out_channels = 1

        self.conv1 = GCNConv(3, 16)
        self.conv2 = GCNConv(16, out_channels)

    def forward(self, x, edge_index, batch=None, edge_label_index=None):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)

        if self.model_config.task_level == ModelTaskLevel.graph:
            x = global_add_pool(x, batch)
        elif self.model_config.task_level == ModelTaskLevel.edge:
            assert edge_label_index is not None
            x = x[edge_label_index[0]] * x[edge_label_index[1]]

        if self.model_config.mode == ModelMode.binary_classification:
            if self.model_config.return_type == ModelReturnType.probs:
                x = x.sigmoid()
        elif self.model_config.mode == ModelMode.multiclass_classification:
            if self.model_config.return_type == ModelReturnType.probs:
                x = x.softmax(dim=-1)
            elif self.model_config.return_type == ModelReturnType.log_probs:
                x = x.log_softmax(dim=-1)

        return x


node_mask_types = [MaskType.attributes, None]
edge_mask_types = [MaskType.object, None]
task_levels = [ModelTaskLevel.node, ModelTaskLevel.edge, ModelTaskLevel.graph]
indices = [1, torch.arange(2)]


def check_explanation(
    explanation: Explanation,
    node_mask_type: Optional[MaskType],
    edge_mask_type: Optional[MaskType],
):
    if node_mask_type == MaskType.attributes:
        assert explanation.node_mask.size() == explanation.x.size()
    elif node_mask_type is None:
        assert 'node_mask' not in explanation

    if edge_mask_type == MaskType.object:
        assert explanation.edge_mask.size() == (explanation.num_edges, )
    elif edge_mask_type is None:
        assert 'edge_mask' not in explanation


@withPackage('captum')
@pytest.mark.parametrize('method', unsupported_methods)
def test_unsupported_methods(method):
    model_config = ModelConfig(mode='regression', task_level='node')

    with pytest.raises(ValueError, match="does not support attribution"):
        Explainer(
            GCN(model_config),
            algorithm=CaptumExplainer(method),
            explanation_type='model',
            edge_mask_type='object',
            node_mask_type='attributes',
            model_config=model_config,
        )


@withPackage('captum')
@pytest.mark.parametrize('method', ['IntegratedGradients'])
@pytest.mark.parametrize('node_mask_type', node_mask_types)
@pytest.mark.parametrize('edge_mask_type', edge_mask_types)
@pytest.mark.parametrize('task_level', task_levels)
@pytest.mark.parametrize('index', indices)
def test_captum_explainer_binary_classification(
    method,
    data,
    node_mask_type,
    edge_mask_type,
    task_level,
    index,
):
    if node_mask_type is None and edge_mask_type is None:
        return

    batch = torch.tensor([0, 0, 1, 1])
    edge_label_index = torch.tensor([[0, 1, 2], [2, 3, 1]])

    model_config = ModelConfig(
        mode='binary_classification',
        task_level=task_level,
        return_type='probs',
    )

    explainer = Explainer(
        GCN(model_config),
        algorithm=CaptumExplainer(method),
        explanation_type='model',
        edge_mask_type=edge_mask_type,
        node_mask_type=node_mask_type,
        model_config=model_config,
    )

    explanation = explainer(
        data.x,
        data.edge_index,
        index=index,
        batch=batch,
        edge_label_index=edge_label_index,
    )
    check_explanation(explanation, node_mask_type, edge_mask_type)


@withPackage('captum')
@pytest.mark.parametrize('method', methods)
@pytest.mark.parametrize('node_mask_type', node_mask_types)
@pytest.mark.parametrize('edge_mask_type', edge_mask_types)
@pytest.mark.parametrize('task_level', task_levels)
@pytest.mark.parametrize('index', indices)
def test_captum_explainer_multiclass_classification(
    method,
    data,
    node_mask_type,
    edge_mask_type,
    task_level,
    index,
):
    if node_mask_type is None and edge_mask_type is None:
        return

    batch = torch.tensor([0, 0, 1, 1])
    edge_label_index = torch.tensor([[0, 1, 2], [2, 3, 1]])

    model_config = ModelConfig(
        mode='multiclass_classification',
        task_level=task_level,
        return_type='probs',
    )

    explainer = Explainer(
        GCN(model_config),
        algorithm=CaptumExplainer(method),
        explanation_type='model',
        edge_mask_type=edge_mask_type,
        node_mask_type=node_mask_type,
        model_config=model_config,
    )

    explanation = explainer(
        data.x,
        data.edge_index,
        index=index,
        batch=batch,
        edge_label_index=edge_label_index,
    )
    check_explanation(explanation, node_mask_type, edge_mask_type)


@withPackage('captum')
@pytest.mark.parametrize(
    'method',
    [m for m in methods if m != 'ShapleyValueSampling'],
)
@pytest.mark.parametrize(
    'node_mask_type',
    [nm for nm in node_mask_types if nm is not None],
)
@pytest.mark.parametrize(
    'edge_mask_type',
    [em for em in edge_mask_types if em is not None],
)
@pytest.mark.parametrize('index', [1, torch.arange(2)])
def test_captum_hetero_data(method, node_mask_type, edge_mask_type, index,
                            hetero_data, hetero_model):

    model_config = ModelConfig(mode='regression', task_level='node')

    explainer = Explainer(
        hetero_model(hetero_data.metadata()),
        algorithm=CaptumExplainer(method),
        edge_mask_type=edge_mask_type,
        node_mask_type=node_mask_type,
        model_config=model_config,
        explanation_type='model',
    )

    explanation = explainer(hetero_data.x_dict, hetero_data.edge_index_dict,
                            index=index)

    explanation.validate(raise_on_error=True)


@withPackage('captum')
@pytest.mark.parametrize('node_mask_type', [
    MaskType.object,
    MaskType.common_attributes,
])
def test_captum_explainer_supports(node_mask_type):
    model_config = ModelConfig(
        mode='multiclass_classification',
        task_level='node',
        return_type='probs',
    )

    with pytest.raises(ValueError, match="not support the given explanation"):
        Explainer(
            GCN(model_config),
            algorithm=CaptumExplainer('IntegratedGradients'),
            edge_mask_type=MaskType.object,
            node_mask_type=node_mask_type,
            model_config=model_config,
            explanation_type='model',
        )
