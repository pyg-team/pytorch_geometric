from typing import Literal, Optional

import pytest
import torch

from torch_geometric.explain import Explainer, HeteroExplanation
from torch_geometric.explain.algorithm import CaptumExplainer
from torch_geometric.explain.config import (
    MaskType,
    ModelConfig,
    ModelMode,
    ModelReturnType,
    ModelTaskLevel,
)
from torch_geometric.nn import HANConv, HGTConv, global_add_pool
from torch_geometric.testing import withPackage
from torch_geometric.transforms import ToSparseTensor

methods = [
    'Saliency',
    'InputXGradient',
    'IntegratedGradients',
    'GuidedBackprop',
    'ShapleyValueSampling',
]

unsupported_methods = [
    'Deconvolution',
    'FeatureAblation',
    'Occlusion',
    'DeepLift',
    'DeepLiftShap',
    'GradientShap',
    'KernelShap',
    'Lime',
]


class HetGraphNet(torch.nn.Module):
    def __init__(self, model_config: ModelConfig, metadata: dict,
                 in_channels: int = -1,
                 base_conv: Literal['han', 'hgt'] = 'han', hid_dim: int = 8):
        super().__init__()

        self.model_config = model_config
        self.hid_dim = hid_dim
        if model_config.mode == ModelMode.multiclass_classification:
            self.out_channels = 7
        else:
            self.out_channels = 1

        if base_conv == 'hgt':
            self.conv = HGTConv(in_channels, self.hid_dim, heads=2,
                                metadata=metadata)
            self.conv2 = HGTConv(self.hid_dim, self.out_channels, heads=1,
                                 metadata=metadata)
        elif base_conv == 'han':
            self.conv = HANConv(in_channels, self.hid_dim, heads=2,
                                metadata=metadata)
            self.conv2 = HANConv(self.hid_dim, self.out_channels, heads=1,
                                 metadata=metadata)
        else:
            raise ValueError(f"Unsupported base conv {base_conv}")
        self.relu = torch.nn.ReLU()

    def forward(self, x_dict, edge_index_dict, batch_info=None,
                edge_label_index=None):
        x_dict_n = x_dict

        x_dict_n = self.conv(x_dict_n, edge_index_dict)

        x_dict_n = {k: self.relu(v) for k, v in x_dict_n.items()}
        x_dict_n = self.conv2(x_dict_n, edge_index_dict)

        if (self.model_config.task_level == ModelTaskLevel.graph):
            x_dict_n = {
                k: global_add_pool(v, batch_info[k])
                for k, v in x_dict_n.items()
            }
            x = torch.stack(list(x_dict_n.values()), dim=-1)
            x = x.sum(dim=-1)
        elif (self.model_config.task_level == ModelTaskLevel.edge):
            assert edge_label_index is not None
            x = [
                x_dict_n[node_type][mask]
                for node_type, mask in edge_label_index.items()
            ]
            x = torch.stack(x).sum(dim=0)
        else:
            # Node classification
            x = x_dict_n['paper']

        if self.model_config.mode == ModelMode.binary_classification:
            if self.model_config.return_type == ModelReturnType.probs:
                x = x.sigmoid()
        elif self.model_config.mode == ModelMode.multiclass_classification:
            if self.model_config.return_type == ModelReturnType.probs:
                x = x.softmax(dim=-1)
            elif self.model_config.return_type == ModelReturnType.log_probs:
                x = x.log_softmax(dim=-1)

        return x


def check_hetero_explanation(
    explanation: HeteroExplanation,
    node_mask_type: Optional[MaskType],
    edge_mask_type: Optional[MaskType],
):

    if node_mask_type == MaskType.attributes:
        for node_type, node_mask in explanation.node_mask_dict.items():
            assert node_mask.size() == explanation.x_dict[node_type].size()
    elif node_mask_type is None:
        with pytest.raises(KeyError, match="Tried to collect"):
            _ = explanation.node_mask_dict

    if edge_mask_type == MaskType.object:
        for edge_type, edge_mask in explanation.edge_mask_dict.items():
            assert edge_mask.size() == (explanation[edge_type].num_edges, )
    elif edge_mask_type is None:
        with pytest.raises(KeyError, match="Tried to collect"):
            _ = explanation.edge_mask_dict


mode_types = [
    'binary_classification', 'multiclass_classification', 'regression'
]
ret_types = ['probs', 'log_probs', 'raw']
node_mask_types = [MaskType.attributes, None]
edge_mask_types = [MaskType.object, None]
task_levels = [ModelTaskLevel.node, ModelTaskLevel.edge, ModelTaskLevel.graph]
indices = [1, torch.arange(2)]

# currently, sparse fails independently of code changes related to this test
# when torch_sparse is installed, so we disable since adding code to handle
# this case explicitly is not a priority as it fails for other reasons
# edge_type = ['dense', 'sparse']
edge_type = ['dense']


# Note: no reason to believe tests will pass IntegratedGradients
# but fail for other methods that are supported, so we just test
# with IntegratedGradients to reduce quantity of tests overall
@withPackage('captum')
@pytest.mark.parametrize('method', ['IntegratedGradients'])
@pytest.mark.parametrize('mode_type', mode_types)
@pytest.mark.parametrize('ret_type', ret_types)
@pytest.mark.parametrize('node_mask_type', node_mask_types)
@pytest.mark.parametrize('edge_mask_type', edge_mask_types)
@pytest.mark.parametrize('task_level', task_levels)
@pytest.mark.parametrize('index', indices)
@pytest.mark.parametrize('conv_type', ['hgt', 'han'])
@pytest.mark.parametrize('edge_type', edge_type)
def test_captum_explainer_special_heterogeneous(
    method,
    mode_type,
    ret_type,
    hetero_data,
    node_mask_type,
    edge_mask_type,
    task_level,
    index,
    conv_type,
    edge_type,
):

    if node_mask_type is None and edge_mask_type is None:
        pytest.skip(
            "No meaning to explain without an edge mask or a node mask")

    # ModelConfig ValueError (as expected)
    if mode_type == 'regression' and ret_type != 'raw':
        pytest.xfail("Non-raw outputs not supported for regression tasks")

    # ModelConfig ValueError
    if mode_type == 'binary_classification' and ret_type == 'log_probs':
        pytest.xfail("Log-prob outputs not supported for binary tasks")

    # CaptumExplainer ValueError
    if mode_type == 'binary_classification' and ret_type == 'raw':
        pytest.xfail(
            "CaptumExplainer does not support raw outputs for binary tasks")

    # ShapleyValueSampling very slow to test
    # was functional on initial set of tests, so disabled here
    if method == 'ShapleyValueSampling':
        pytest.skip(
            "ShapleyValueSampling testing takes too long, so skipping...")

    batch_info = {
        'paper': torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
        'author': torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    }
    edge_label_index = {
        'paper': torch.tensor([0, 1, 2]),
        'author': torch.tensor([2, 3, 1])
    }

    model_config = ModelConfig(
        mode=mode_type,
        task_level=task_level,
        return_type=ret_type,
    )

    model = HetGraphNet(model_config, hetero_data.metadata(),
                        base_conv=conv_type)

    explainer = Explainer(
        model,
        algorithm=CaptumExplainer(method),
        explanation_type='model',
        edge_mask_type=edge_mask_type,
        node_mask_type=node_mask_type,
        model_config=model_config,
    )

    if edge_type == 'dense':
        explanation = explainer(
            hetero_data.x_dict,
            hetero_data.edge_index_dict,
            index=index,
            batch_info=batch_info,
            edge_label_index=edge_label_index,
        )

        check_hetero_explanation(explanation, node_mask_type, edge_mask_type)

    elif edge_type == 'sparse':
        sparse_transform = ToSparseTensor()
        hetero_data = sparse_transform(hetero_data)

        # Fail on sparse edge masking
        if edge_mask_type == MaskType.object:
            with pytest.raises(ValueError,
                               match="Sparse edge index not supported"):
                explanation = explainer(
                    hetero_data.x_dict,
                    hetero_data.adj_t_dict,
                    index=index,
                    batch_info=batch_info,
                    edge_label_index=edge_label_index,
                )
        else:
            explanation = explainer(
                hetero_data.x_dict,
                hetero_data.adj_t_dict,
                index=index,
                batch_info=batch_info,
                edge_label_index=edge_label_index,
            )

            check_hetero_explanation(explanation, node_mask_type,
                                     edge_mask_type)
