"""WHG-Retriever demo utilities.

Provides:
- WarehouseConversationSystem: text-based interface for a warehouse graph
- demo_whg_retriever(): command-line demo
- test_pyg_integration(): lightweight self-test
"""

from __future__ import annotations

import time

import torch

# Import our model components
from .whg_model import WarehouseGRetriever

# Optional PyG imports with graceful fallback
try:
    from torch_geometric.data import HeteroData
    HETERO_DATA_AVAILABLE = True
except ImportError:
    HETERO_DATA_AVAILABLE = False
    HeteroData = type(None)  # type: ignore


class WarehouseConversationSystem:
    """Text-based interface for querying a warehouse graph.

    Accepts natural language questions and returns analysis using
    PyG GAT and multi-task classification heads.

    Args:
        hetero_data (Union[HeteroData, torch.Tensor]): Warehouse graph data.
            Can be PyG HeteroData or tensor with node features.
        edge_index (torch.Tensor, optional): Edge connectivity matrix
            if hetero_data is a tensor. Default: None.
        llm_model (str, optional): LLM model name for advanced responses.
            Default: None.

    Examples:
        >>> from torch_geometric.datasets.relbench import (
        ...     create_relbench_hetero_data
        ... )
        >>> data = create_relbench_hetero_data('rel-trial')
        >>> warehouse = WarehouseConversationSystem(data)
        >>> response = warehouse.ask("What is the structure?")
        >>> print(response)
    """
    def __init__(
        self,
        hetero_data: HeteroData | torch.Tensor,
        edge_index: torch.Tensor | None = None,
        llm_model: str | None = None,
    ) -> None:
        """Initialize using PyG components.

        Args:
            hetero_data: HeteroData or tensor with node features
            edge_index: Edge connections (if hetero_data is tensor)
            llm_model: Optional LLM model name for advanced responses
        """
        if HETERO_DATA_AVAILABLE and isinstance(hetero_data, HeteroData):
            data = hetero_data.to_homogeneous()
            self.x = data.x
            self.edge_index = data.edge_index

            self.node_metadata = {}
            offset = 0
            for node_type in hetero_data.node_types:
                num_nodes = hetero_data[node_type].num_nodes
                for i in range(num_nodes):
                    self.node_metadata[offset + i] = {
                        'type': node_type,
                        'layer': self._infer_layer(node_type),
                    }
                offset += num_nodes
        else:
            if not isinstance(hetero_data, torch.Tensor):
                raise ValueError(
                    "hetero_data must be a tensor when not using HeteroData")
            self.x = hetero_data
            if edge_index is None:
                raise ValueError(
                    "edge_index must be provided when hetero_data is a tensor")
            self.edge_index = edge_index
            self.node_metadata = {
                i: {
                    'type': f'table_{i}',
                    'layer': 'unknown'
                }
                for i in range(hetero_data.shape[0])
            }

        # Ensure tensors are properly initialized
        assert self.x is not None, "Node features not initialized"
        assert self.edge_index is not None, "Edge index not initialized"

        self.warehouse_ai = WarehouseGRetriever(input_dim=self.x.shape[1],
                                                hidden_dim=128,
                                                llm_model=llm_model)

    def _infer_layer(self, node_type: str) -> str:
        """Infer warehouse layer from node type."""
        node_type_lower = node_type.lower()
        if any(term in node_type_lower for term in ['source', 'raw', 'src']):
            return 'source'
        elif any(term in node_type_lower for term in ['staging', 'stg']):
            return 'staging'
        elif any(term in node_type_lower for term in ['enriched', 'enrich']):
            return 'enriched'
        elif any(term in node_type_lower for term in ['mart', 'dm']):
            return 'mart'
        return 'unknown'

    def ask(self, question: str) -> str:
        """Ask warehouse question using PyG G-Retriever pipeline.

        Processes natural language questions about warehouse structure,
        connectivity, and data quality using neural graph analysis.

        Args:
            question (str): Natural language question about the warehouse.

        Returns:
            str: Intelligent analysis response based on graph neural network
                predictions and warehouse-specific insights.

        Examples:
            >>> response = warehouse.ask("What is the data lineage?")
            >>> response = warehouse.ask("Are there any data silos?")
            >>> response = warehouse.ask("How is the data quality?")
        """
        start_time = time.time()

        # MyPy type narrowing
        assert self.x is not None
        assert self.edge_index is not None

        relevant_nodes = self.warehouse_ai.retrieve_relevant_nodes(
            question, self.x, self.node_metadata)

        analysis = self.warehouse_ai.analyze_warehouse_tasks(
            relevant_nodes, self.x, self.edge_index)

        response = self.warehouse_ai.generate_warehouse_response(
            question, analysis)

        response_time = time.time() - start_time
        response += (f'\n\nAnalysis completed in {response_time:.3f}s using '
                     'PyG G-Retriever infrastructure.')

        return response


# Alias for backward compatibility
WHGConversationSystem = WarehouseConversationSystem


def demo_whg_retriever() -> None:
    """Command-line demo of WHG-Retriever."""
    print('WHG-RETRIEVER DEMO (Warehouse G-Retriever using PyG)')
    print('=' * 70)

    print('Testing PyG component integration...')

    print('Creating synthetic warehouse data...')
    x = torch.randn(40, 384)  # 40 tables with 384-dim features

    edges = []
    # Source to staging (0-9 -> 10-19)
    for i in range(10):
        for j in range(10, 20):
            if torch.rand(1) > 0.7:
                edges.append([i, j])

    # Staging to mart (10-19 -> 20-29)
    for i in range(10, 20):
        for j in range(20, 30):
            if torch.rand(1) > 0.6:
                edges.append([i, j])

    # Some isolated silos (30-39)
    for i in range(30, 35):
        if torch.rand(1) > 0.9:
            j = int(torch.randint(0, 30, (1, )).item())
            edges.append([i, j])

    edge_index = (torch.tensor(edges).t().contiguous()
                  if edges else torch.empty((2, 0), dtype=torch.long))

    print(f'   • {x.shape[0]} warehouse tables')
    print(f'   • {edge_index.shape[1]} connections')

    print('Initializing PyG-based warehouse system...')
    warehouse = WarehouseConversationSystem(x, edge_index)

    demo_questions = [
        'What is the overall structure of this data warehouse?',
        'Are there any isolated data silos that need attention?',
        'How is the data quality across the warehouse?',
        'Show me information about the mart layer tables.',
        'What source tables are feeding this warehouse?',
        'Identify any connectivity issues in the warehouse.',
    ]

    print('\nCOMPREHENSIVE WAREHOUSE CONVERSATIONS:')
    print('-' * 60)

    for i, question in enumerate(demo_questions, 1):
        print(f'\n{i}. Human: {question}')

        try:
            response = warehouse.ask(question)

            lines = response.split('\n')
            key_insights = []

            for line in lines:
                if '•' in line:
                    key_insights.append(line.strip())

            print('   WHG-Retriever Analysis:')
            for insight in key_insights[:4]:  # Show top 4 insights
                if insight:
                    print(f'      {insight}')

            if 'PyG' in response:
                print('      Using PyG infrastructure')

        except Exception as e:
            print(f'   Error: {e}')

        print('   ' + '-' * 40)

    print('\nWHG-Retriever warehouse demo complete!')

    print('\nWHG-RETRIEVER COMPONENT SUMMARY:')
    print('   Graph Neural Network: PyG GAT')
    text_encoding = ("PyG SBERT" if warehouse.warehouse_ai.sentence_transformer
                     else "Keyword Fallback")
    print(f'   Text Encoding: {text_encoding}')
    print('   Warehouse Tasks: Custom Extensions')
    llm_integration = ("PyG LLM"
                       if warehouse.warehouse_ai.llm else "Enhanced Templates")
    print(f'   LLM Integration: {llm_integration}')
    print('   Data Integration: PyG HeteroData Support')


# Backward compatibility alias
demo_warehouse_g_retriever = demo_whg_retriever


def test_pyg_integration() -> None:
    """Test PyG component integration."""
    print('\nTESTING PyG INTEGRATION')
    print('=' * 40)

    try:
        print('Testing PyG GAT integration...')
        warehouse_ai = WarehouseGRetriever(input_dim=384, hidden_dim=128)

        gat_module = warehouse_ai.gnn.__class__.__module__
        print(f'   GAT module: {gat_module}')
        gat_status = ("Available"
                      if "torch_geometric" in gat_module else "Unavailable")
        print(f'   PyG GAT: {gat_status}')

        x = torch.randn(20, 384)
        edge_index = torch.randint(0, 20, (2, 30))

        embeddings = warehouse_ai.encode_graph(x, edge_index)
        print(f'   Graph encoding: {embeddings.shape}')

        task_outputs = warehouse_ai.warehouse_tasks(embeddings)
        print(f'   Lineage predictions: {task_outputs["lineage"].shape}')
        print(f'   Silo predictions: {task_outputs["silo"].shape}')
        print(f'   Anomaly predictions: {task_outputs["anomaly"].shape}')

        node_metadata = {
            i: {
                'type': f'table_{i}',
                'layer': 'unknown'
            }
            for i in range(20)
        }
        relevant_nodes = warehouse_ai.retrieve_relevant_nodes(
            'What are the source tables?', x, node_metadata)
        print(f'   Node retrieval: {len(relevant_nodes)} nodes')

        analysis = warehouse_ai.analyze_warehouse_tasks(
            relevant_nodes, x, edge_index)
        print(f'   Warehouse analysis: {analysis["total_analyzed"]} nodes')

        print('All PyG integration tests passed!')

    except Exception as e:
        print(f'PyG integration test failed: {e}')


if __name__ == '__main__':
    demo_warehouse_g_retriever()
    test_pyg_integration()
