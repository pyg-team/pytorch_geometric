import pytest
import torch

from torch_geometric.explain import Explainer, HeteroExplanation, PGExplainer
from torch_geometric.explain.config import (
    ExplanationType,
    ModelConfig,
    ModelMode,
    ModelReturnType,
    ModelTaskLevel,
)
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.testing import withCUDA


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
        return x


@withCUDA
@pytest.mark.parametrize('mode', [
    ModelMode.binary_classification,
    ModelMode.multiclass_classification,
    ModelMode.regression,
])
def test_pg_explainer_node(device, check_explanation, mode):
    x = torch.randn(8, 3, device=device)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
        [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6],
    ], device=device)

    if mode == ModelMode.binary_classification:
        target = torch.randint(2, (x.size(0), ), device=device)
    elif mode == ModelMode.multiclass_classification:
        target = torch.randint(7, (x.size(0), ), device=device)
    elif mode == ModelMode.regression:
        target = torch.randn((x.size(0), 1), device=device)

    model_config = ModelConfig(mode=mode, task_level='node', return_type='raw')

    model = GCN(model_config).to(device)

    explainer = Explainer(
        model=model,
        algorithm=PGExplainer(epochs=2).to(device),
        explanation_type='phenomenon',
        edge_mask_type='object',
        model_config=model_config,
    )

    with pytest.raises(ValueError, match="not yet fully trained"):
        explainer(x, edge_index, target=target)

    explainer.algorithm.reset_parameters()
    for epoch in range(2):
        for index in range(x.size(0)):
            loss = explainer.algorithm.train(epoch, model, x, edge_index,
                                             target=target, index=index)
            assert loss >= 0.0

    explanation = explainer(x, edge_index, target=target, index=0)

    check_explanation(explanation, None, explainer.edge_mask_type)


@withCUDA
@pytest.mark.parametrize('mode', [
    ModelMode.binary_classification,
    ModelMode.multiclass_classification,
    ModelMode.regression,
])
def test_pg_explainer_graph(device, check_explanation, mode):
    x = torch.randn(8, 3, device=device)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
        [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6],
    ], device=device)

    if mode == ModelMode.binary_classification:
        target = torch.randint(2, (1, ), device=device)
    elif mode == ModelMode.multiclass_classification:
        target = torch.randint(7, (1, ), device=device)
    elif mode == ModelMode.regression:
        target = torch.randn((1, 1), device=device)

    model_config = ModelConfig(mode=mode, task_level='graph',
                               return_type='raw')

    model = GCN(model_config).to(device)

    explainer = Explainer(
        model=model,
        algorithm=PGExplainer(epochs=2).to(device),
        explanation_type='phenomenon',
        edge_mask_type='object',
        model_config=model_config,
    )

    with pytest.raises(ValueError, match="not yet fully trained"):
        explainer(x, edge_index, target=target)

    explainer.algorithm.reset_parameters()
    for epoch in range(2):
        loss = explainer.algorithm.train(epoch, model, x, edge_index,
                                         target=target)
        assert loss >= 0.0

    explanation = explainer(x, edge_index, target=target)

    check_explanation(explanation, None, explainer.edge_mask_type)


@withCUDA
@pytest.mark.parametrize('mode', [
    ModelMode.binary_classification,
    ModelMode.multiclass_classification,
    ModelMode.regression,
])
@pytest.mark.parametrize('task_level', [
    ModelTaskLevel.node,
    ModelTaskLevel.graph,
])
def test_pg_explainer_hetero(device, hetero_data, hetero_model,
                             check_explanation_hetero, mode, task_level):
    # Move data to device
    # for key, value in hetero_data.x_dict.items():
    #     hetero_data.x_dict[key] = value.to(device)
    # for key, value in hetero_data.edge_index_dict.items():
    #     hetero_data.edge_index_dict[key] = value.to(device)
    hetero_data = hetero_data.to(device)

    # Prepare target based on mode and task level
    if task_level == ModelTaskLevel.node:
        # num_nodes = hetero_data['paper'].x.size(0)
        # if mode == ModelMode.binary_classification:
        #     target = torch.randint(2, (num_nodes,), device=device)
        # elif mode == ModelMode.multiclass_classification:
        #     target = torch.randint(7, (num_nodes,), device=device)
        # elif mode == ModelMode.regression:
        #     target = torch.randn((num_nodes, 1), device=device)
        index = 0  # For node-level explanation
    else:  # graph-level
        # if mode == ModelMode.binary_classification:
        #     target = torch.randint(2, (1,), device=device)
        # elif mode == ModelMode.multiclass_classification:
        #     target = torch.randint(7, (1,), device=device)
        # elif mode == ModelMode.regression:
        #     target = torch.randn((1, 1), device=device)
        index = None  # For graph-level explanation

    # with torch.no_grad():
    #     target = model(hetero_data.x_dict,
    #                     hetero_data.edge_index_dict).argmax(-1)

    # Create model config
    model_config = ModelConfig(
        mode=mode,
        task_level=task_level,
        return_type=ModelReturnType.raw,
    )

    # Create and initialize model
    metadata = hetero_data.metadata()
    model = hetero_model(metadata, model_config).to(device)

    with torch.no_grad():
        raw_output = model(hetero_data.x_dict, hetero_data.edge_index_dict)
        if mode == ModelMode.multiclass_classification:
            # For multiclass, use class indices (long tensor)
            target = raw_output.argmax(dim=-1)
        elif mode == ModelMode.binary_classification:
            # For binary, convert to binary targets (long tensor)
            target = (raw_output > 0).long()
        else:  # regression
            # For regression, use raw outputs (float tensor)
            target = raw_output.float()

    # Create explainer
    explainer = Explainer(
        model=model,
        algorithm=PGExplainer(epochs=2).to(device),
        explanation_type=ExplanationType.phenomenon,
        edge_mask_type='object',
        model_config=model_config,
    )

    # Should raise error when not fully trained
    with pytest.raises(ValueError, match="not yet fully trained"):
        explainer(
            hetero_data.x_dict,
            hetero_data.edge_index_dict,
            target=target,
            index=index if task_level == ModelTaskLevel.node else None,
        )

    # Train the explainer
    explainer.algorithm.reset_parameters()
    for epoch in range(2):
        if task_level == ModelTaskLevel.node:
            # For node-level, train on a single node
            loss = explainer.algorithm.train(
                epoch,
                model,
                hetero_data.x_dict,
                hetero_data.edge_index_dict,
                target=target,
                index=index,
            )
        else:
            # For graph-level, train on the whole graph
            loss = explainer.algorithm.train(
                epoch,
                model,
                hetero_data.x_dict,
                hetero_data.edge_index_dict,
                target=target,
            )
        assert isinstance(loss, float)

    # Get explanation
    explanation = explainer(
        hetero_data.x_dict,
        hetero_data.edge_index_dict,
        target=target,
        index=index if task_level == ModelTaskLevel.node else None,
    )

    # Check if the explanation is valid
    assert isinstance(explanation, HeteroExplanation)
    # Run through the standard explanation checker
    check_explanation_hetero(explanation, None, explainer.edge_mask_type,
                             hetero_data)


def test_pg_explainer_supports():
    # Test unsupported model task level:
    with pytest.raises(ValueError, match="not support the given explanation"):
        model_config = ModelConfig(
            mode='binary_classification',
            task_level='edge',
            return_type='raw',
        )
        Explainer(
            model=GCN(model_config),
            algorithm=PGExplainer(epochs=2),
            explanation_type='phenomenon',
            edge_mask_type='object',
            model_config=model_config,
        )

    # Test unsupported explanation type:
    with pytest.raises(ValueError, match="not support the given explanation"):
        model_config = ModelConfig(
            mode='binary_classification',
            task_level='node',
            return_type='raw',
        )
        Explainer(
            model=GCN(model_config),
            algorithm=PGExplainer(epochs=2),
            explanation_type='model',
            edge_mask_type='object',
            model_config=model_config,
        )

    # Test unsupported node mask:
    with pytest.raises(ValueError, match="not support the given explanation"):
        model_config = ModelConfig(
            mode='binary_classification',
            task_level='node',
            return_type='raw',
        )
        Explainer(
            model=GCN(model_config),
            algorithm=PGExplainer(epochs=2),
            explanation_type='model',
            node_mask_type='object',
            edge_mask_type='object',
            model_config=model_config,
        )
