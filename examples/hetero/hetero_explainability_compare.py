import torch
import numpy as np
import random
from torch_geometric.explain import Explainer, GNNExplainer, HeteroExplanation
from torch_geometric.explain.config import (
    ExplanationType,
    MaskType,
    ModelConfig,
    ModelMode,
    ModelReturnType,
    ModelTaskLevel,
)
from torch_geometric.data import HeteroData
from torch_geometric.testing import get_random_edge_index
from typing import Optional
from torch_geometric.nn import to_hetero, SAGEConv
import copy

# Set random seeds for reproducibility
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class GraphSAGE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), 32)
        self.conv2 = SAGEConv((-1, -1), 32)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

class HeteroSAGE(torch.nn.Module):
    def __init__(self, metadata, model_config: Optional[ModelConfig] = None):
        super().__init__()
        self.model_config = model_config
        self.graph_sage = to_hetero(GraphSAGE(), metadata, debug=False)

        # Determine output channels based on model_config
        out_channels = 1
        if (model_config
                and model_config.mode == ModelMode.multiclass_classification):
            out_channels = 7

        self.lin = torch.nn.Linear(32, out_channels)

    def forward(self, x_dict, edge_index_dict) -> torch.Tensor:
        x = self.lin(self.graph_sage(x_dict, edge_index_dict)['paper'])

        # Apply transformations based on model_config if available
        if self.model_config:
            if self.model_config.mode == ModelMode.binary_classification:
                if self.model_config.return_type == ModelReturnType.probs:
                    x = x.sigmoid()
            elif self.model_config.mode == ModelMode.multiclass_classification:
                if self.model_config.return_type == ModelReturnType.probs:
                    x = x.softmax(dim=-1)
                elif (self.model_config.return_type ==
                      ModelReturnType.log_probs):
                    x = x.log_softmax(dim=-1)

        return x
    

def create_hetero_data():
    # Create a simple heterogeneous graph
    data = HeteroData()
    feature_size = 8
    data['paper'].x = torch.randn(8, feature_size)
    data['author'].x = torch.randn(10, feature_size)

    data['paper', 'paper'].edge_index = get_random_edge_index(8, 8, 10)
    data['paper', 'paper'].edge_attr = torch.randn(10, feature_size)
    data['paper', 'author'].edge_index = get_random_edge_index(8, 10, 10)
    data['paper', 'author'].edge_attr = torch.randn(10, feature_size)
    data['author', 'paper'].edge_index = get_random_edge_index(10, 8, 10)
    data['author', 'paper'].edge_attr = torch.randn(10, feature_size)

    return data

def create_hetero_data_with_one_node_type(hetero_data: HeteroData, node_type: str = 'paper'):
    hetero_data_with_one_node_type = copy.deepcopy(hetero_data)
    # Keep only the specified node type and remove others
    node_types_to_remove = [nt for nt in hetero_data.node_types if nt != node_type]
    for nt in node_types_to_remove:
        del hetero_data_with_one_node_type[nt]

    # Remove any edge types that don't connect to the specified node type
    edge_types_to_remove = []
    for edge_type in hetero_data_with_one_node_type.edge_types:
        src, _, dst = edge_type
        if src != node_type or dst != node_type:
            edge_types_to_remove.append(edge_type)
    
    for et in edge_types_to_remove:
        del hetero_data_with_one_node_type[et]
    return hetero_data_with_one_node_type

def create_hetero_explanation(
    hetero_data: HeteroData,
    model_config: ModelConfig,
    node_mask_type: MaskType = MaskType.object,
    edge_mask_type: MaskType = MaskType.object,
    explanation_type: ExplanationType = ExplanationType.model,
    index: Optional[int] = None,
) -> HeteroExplanation:
    """Create explanations for a heterogeneous graph using GNNExplainer.

    Args:
        hetero_data (HeteroData): The heterogeneous graph data containing
            node features and edge indices.
        model_config (ModelConfig): Configuration for the model including
            mode, task level, and return type.
        node_mask_type (MaskType, optional): Type of node mask to use.
            Defaults to MaskType.object.
        edge_mask_type (MaskType, optional): Type of edge mask to use.
            Defaults to MaskType.object.
        explanation_type (ExplanationType, optional): Type of explanation
            to generate. Defaults to ExplanationType.model.
        index (Optional[int], optional): Index of the node to explain. If
            None, explains all nodes. Defaults to None.

    Returns:
        HeteroExplanation: The generated explanation containing node and edge masks.
    """
    # Create and initialize model
    metadata = hetero_data.metadata()
    model = HeteroSAGE(metadata, model_config)

    target = None
    # Create explainer
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=2),
        explanation_type=explanation_type,
        node_mask_type=node_mask_type,
        edge_mask_type=edge_mask_type,
        model_config=model_config,
    )

    # Generate explanation
    explanation = explainer(
        hetero_data.x_dict,
        hetero_data.edge_index_dict,
        target=target,
        index=index,
    )
    assert isinstance(explanation, HeteroExplanation)
    return explanation

def main():
    # Set seed for reproducibility
    set_seed(42)

    # Set up parameters
    node_mask_type = MaskType.object
    edge_mask_type = MaskType.object
    explanation_type = ExplanationType.model
    task_level = ModelTaskLevel.node
    return_type = ModelReturnType.log_probs
    index = None

    model_config = ModelConfig(
        mode='multiclass_classification',
        task_level=task_level,
        return_type=return_type,
    )

    # Create heterogeneous data with more than one node and edge type
    hetero_data = create_hetero_data()
    # Create heterogeneous data with one node and edge type
    hetero_data_with_one_node_type = create_hetero_data_with_one_node_type(hetero_data, node_type='paper')

    hetero_explanation_with_one_node_type = create_hetero_explanation(
        hetero_data=hetero_data_with_one_node_type,
        model_config=model_config,
        node_mask_type=node_mask_type,
        edge_mask_type=edge_mask_type,
        explanation_type=explanation_type,
        index=index,
    )

    # Generate explanation
    hetero_explanation = create_hetero_explanation(
        hetero_data=hetero_data,
        model_config=model_config,
        node_mask_type=node_mask_type,
        edge_mask_type=edge_mask_type,
        explanation_type=explanation_type,
        index=index,
    )

    # Log comparison of both explanations
    print("\nComparison of Heterogeneous Explanations:")
    print("=========================================")
    
    print("\n1. Full Heterogeneous Graph Explanation:")
    print("----------------------------------------")
    print("Node Masks:")
    for node_type, mask in hetero_explanation.node_mask_dict.items():
        print(f"  {node_type}: shape={mask.shape}, mean={mask.mean():.4f}, std={mask.std():.4f}")
    
    print("\nEdge Masks:")
    for edge_type, mask in hetero_explanation.edge_mask_dict.items():
        print(f"  {edge_type}: shape={mask.shape}, mean={mask.mean():.4f}, std={mask.std():.4f}")

    print("\n2. Single Node Type Graph Explanation (Paper only):")
    print("------------------------------------------------")
    print("Node Masks:")
    for node_type, mask in hetero_explanation_with_one_node_type.node_mask_dict.items():
        print(f"  {node_type}: shape={mask.shape}, mean={mask.mean():.4f}, std={mask.std():.4f}")
    
    print("\nEdge Masks:")
    for edge_type, mask in hetero_explanation_with_one_node_type.edge_mask_dict.items():
        print(f"  {edge_type}: shape={mask.shape}, mean={mask.mean():.4f}, std={mask.std():.4f}")

    # Compare paper-related masks between both explanations
    print("\n3. Comparison of Paper-related Masks:")
    print("-------------------------------------")
    if 'paper' in hetero_explanation.node_mask_dict and 'paper' in hetero_explanation_with_one_node_type.node_mask_dict:
        paper_mask_full = hetero_explanation.node_mask_dict['paper']
        paper_mask_single = hetero_explanation_with_one_node_type.node_mask_dict['paper']
        print(f"Paper Node Masks:")
        print(f"  Full graph: mean={paper_mask_full.mean():.4f}, std={paper_mask_full.std():.4f}")
        print(f"  Single type: mean={paper_mask_single.mean():.4f}, std={paper_mask_single.std():.4f}")
        print(f"  Mean difference: {abs(paper_mask_full.mean() - paper_mask_single.mean()):.4f}")

    if ('paper', 'to', 'paper') in hetero_explanation.edge_mask_dict and ('paper', 'to', 'paper') in hetero_explanation_with_one_node_type.edge_mask_dict:
        paper_edge_mask_full = hetero_explanation.edge_mask_dict[('paper', 'to', 'paper')]
        paper_edge_mask_single = hetero_explanation_with_one_node_type.edge_mask_dict[('paper', 'to', 'paper')]
        print(f"\nPaper-Paper Edge Masks:")
        print(f"  Full graph: mean={paper_edge_mask_full.mean():.4f}, std={paper_edge_mask_full.std():.4f}")
        print(f"  Single type: mean={paper_edge_mask_single.mean():.4f}, std={paper_edge_mask_single.std():.4f}")
        print(f"  Mean difference: {abs(paper_edge_mask_full.mean() - paper_edge_mask_single.mean()):.4f}")

if __name__ == "__main__":
    main() 