import pytest
import torch

from torch_geometric.explain import (AttentionExplainer, Explainer,
                                     HeteroExplanation)
from torch_geometric.explain.config import (ExplanationType, MaskType,
                                            ModelConfig, ModelMode)
from torch_geometric.nn import (AttentiveFP, GATConv, GATv2Conv,
                                TransformerConv, to_hetero)
from torch_geometric.nn.conv import HeteroConv


class AttentionGNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GATConv(3, 16, heads=4)
        self.conv2 = GATv2Conv(4 * 16, 16, heads=2)
        self.conv3 = TransformerConv(2 * 16, 7, heads=1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        return x


class HeteroAttentionGNN(torch.nn.Module):
    def __init__(self, metadata, model_config=None):
        super().__init__()
        self.model_config = model_config

        # Create a single BaseGNN that uses all three attention mechanisms
        class BaseGNN(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # Use different attention mechanisms in sequence
                self.conv1 = GATConv((-1, -1), 16, heads=2,
                                     add_self_loops=False)
                self.conv2 = GATv2Conv((-1, -1), 16, heads=2,
                                       add_self_loops=False)
                self.conv3 = TransformerConv((-1, -1), 32, heads=1)

            def forward(self, x, edge_index):
                x = self.conv1(x, edge_index).relu()
                x = self.conv2(x, edge_index).relu()
                x = self.conv3(x, edge_index)
                return x

        # Convert to heterogeneous model with a single to_hetero call
        self.gnn = to_hetero(BaseGNN(), metadata, debug=False)

        # Output dimension based on model config
        out_channels = 7 if (model_config and model_config.mode
                             == ModelMode.multiclass_classification) else 1
        self.lin = torch.nn.Linear(32, out_channels)

    def forward(self, x_dict, edge_index_dict, **kwargs):
        # Process through the heterogeneous GNN
        out_dict = self.gnn(x_dict, edge_index_dict)

        # Project paper node embeddings for classification/regression
        x = self.lin(out_dict['paper'])

        # Apply appropriate output transformation based on model config
        if self.model_config:
            if self.model_config.mode == ModelMode.binary_classification:
                if self.model_config.return_type == 'probs':
                    x = x.sigmoid()
            elif self.model_config.mode == ModelMode.multiclass_classification:
                if self.model_config.return_type == 'probs':
                    x = x.softmax(dim=-1)
                elif self.model_config.return_type == 'log_probs':
                    x = x.log_softmax(dim=-1)

        return x


class HeteroConvAttentionGNN(torch.nn.Module):
    def __init__(self, metadata, model_config=None):
        super().__init__()
        self.model_config = model_config

        # Determine output channels based on model_config
        self.out_channels = 1
        if (model_config
                and model_config.mode == ModelMode.multiclass_classification):
            self.out_channels = 7

        # Initialize node type-specific layers
        self.lin_dict = torch.nn.ModuleDict()
        self.initialized = False

        # Create a dictionary of attention-based convolutions for each edge
        # type
        conv_dict = {}
        for edge_type in metadata[1]:  # metadata[1] contains edge types
            src_type, _, dst_type = edge_type
            if src_type == dst_type:
                # For same node type, use GATConv with add_self_loops=False
                # Use concat=False to avoid dimension issues
                conv_dict[edge_type] = GATConv(
                    (-1, -1), 32, heads=2, add_self_loops=False, concat=False)
            else:
                # For different node types, use GATv2Conv with
                # add_self_loops=False Use concat=False to avoid dimension
                # issues
                conv_dict[edge_type] = GATv2Conv(
                    (-1, -1), 32, heads=2, add_self_loops=False, concat=False)

        # Create the HeteroConv layer
        self.conv = HeteroConv(conv_dict, aggr='sum')

        # Output layer will be initialized in forward pass
        self.out_lin = None

    def _initialize_layers(self, x_dict):
        """Initialize layers with correct dimensions when we first see the
        data.
        """
        if not self.initialized:
            # Initialize input projections
            for node_type, x in x_dict.items():
                in_channels = x.size(-1)
                self.lin_dict[node_type] = torch.nn.Linear(in_channels,
                                                           32).to(x.device)

            # Initialize output projection
            self.out_lin = torch.nn.Linear(32, self.out_channels).to(
                x_dict['paper'].device)

            self.initialized = True

    def forward(self, x_dict, edge_index_dict):
        # Initialize layers if not done yet
        self._initialize_layers(x_dict)

        # Apply node type-specific transformations
        h_dict = {}
        for node_type, x in x_dict.items():
            h_dict[node_type] = self.lin_dict[node_type](x).relu_()

        # Apply heterogeneous convolution
        out_dict = self.conv(h_dict, edge_index_dict)

        # Final transformation for paper nodes
        out = self.out_lin(out_dict['paper'])

        # Apply transformations based on model_config if available
        if self.model_config:
            if self.model_config.mode == ModelMode.binary_classification:
                if self.model_config.return_type == 'probs':
                    out = out.sigmoid()
            elif self.model_config.mode == ModelMode.multiclass_classification:
                if self.model_config.return_type == 'probs':
                    out = out.softmax(dim=-1)
                elif self.model_config.return_type == 'log_probs':
                    out = out.log_softmax(dim=-1)

        return out


x = torch.randn(8, 3)
edge_index = torch.tensor([
    [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
    [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6],
])
edge_attr = torch.randn(edge_index.size(1), 5)
batch = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2])


@pytest.mark.parametrize('index', [None, 2, torch.arange(3)])
def test_attention_explainer(index, check_explanation):
    explainer = Explainer(
        model=AttentionGNN(),
        algorithm=AttentionExplainer(),
        explanation_type='model',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='raw',
        ),
    )

    explanation = explainer(x, edge_index, index=index)
    check_explanation(explanation, None, explainer.edge_mask_type)


@pytest.mark.parametrize('explanation_type', [e for e in ExplanationType])
@pytest.mark.parametrize('node_mask_type', [m for m in MaskType])
def test_attention_explainer_supports(explanation_type, node_mask_type):
    with pytest.raises(ValueError, match="not support the given explanation"):
        Explainer(
            model=AttentionGNN(),
            algorithm=AttentionExplainer(),
            explanation_type=explanation_type,
            node_mask_type=node_mask_type,
            edge_mask_type='object',
            model_config=dict(
                mode='multiclass_classification',
                task_level='node',
                return_type='raw',
            ),
        )


def test_attention_explainer_attentive_fp(check_explanation):
    model = AttentiveFP(3, 16, 1, edge_dim=5, num_layers=2, num_timesteps=2)

    explainer = Explainer(
        model=model,
        algorithm=AttentionExplainer(),
        explanation_type='model',
        edge_mask_type='object',
        model_config=dict(
            mode='binary_classification',
            task_level='node',
            return_type='raw',
        ),
    )

    explanation = explainer(x, edge_index, edge_attr=edge_attr, batch=batch)
    check_explanation(explanation, None, explainer.edge_mask_type)


@pytest.mark.parametrize('index', [None, 2, torch.arange(3)])
def test_attention_explainer_hetero(index, hetero_data,
                                    check_explanation_hetero):
    # Create model configuration
    model_config = ModelConfig(
        mode='multiclass_classification',
        task_level='node',
        return_type='raw',
    )

    # Get metadata from hetero_data
    metadata = hetero_data.metadata()

    # Create the hetero attention model
    model = HeteroAttentionGNN(metadata, model_config)

    # Create the explainer
    explainer = Explainer(
        model=model,
        algorithm=AttentionExplainer(),
        explanation_type='model',
        edge_mask_type='object',
        model_config=model_config,
    )

    # Generate the explanation
    explanation = explainer(
        hetero_data.x_dict,
        hetero_data.edge_index_dict,
        index=index,
    )

    # Check that the explanation is correct
    assert isinstance(explanation, HeteroExplanation)
    check_explanation_hetero(explanation, None, explainer.edge_mask_type,
                             hetero_data)


@pytest.mark.parametrize('index', [None, 2, torch.arange(3)])
def test_attention_explainer_hetero_conv(index, hetero_data,
                                         check_explanation_hetero):
    """Test AttentionExplainer with HeteroConv using attention-based layers."""
    # Create model configuration
    model_config = ModelConfig(
        mode='multiclass_classification',
        task_level='node',
        return_type='raw',
    )

    # Get metadata from hetero_data
    metadata = hetero_data.metadata()

    # Create the hetero conv attention model
    model = HeteroConvAttentionGNN(metadata, model_config)

    # Create the explainer
    explainer = Explainer(
        model=model,
        algorithm=AttentionExplainer(),
        explanation_type='model',
        edge_mask_type='object',
        model_config=model_config,
    )

    # Generate the explanation
    explanation = explainer(
        hetero_data.x_dict,
        hetero_data.edge_index_dict,
        index=index,
    )

    # Check that the explanation is correct
    assert isinstance(explanation, HeteroExplanation)
    check_explanation_hetero(explanation, None, explainer.edge_mask_type,
                             hetero_data)
