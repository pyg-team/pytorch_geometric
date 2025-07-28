"""Warehouse Intelligence Demo - Out of the Box Usage.

Demo of PyTorch Geometric warehouse intelligence with RelBench data.
Includes lineage detection, silo analysis, and quality assessment.

Usage:
    python examples/llm/whg_demo.py

This demo uses the complete implementations from:
- torch_geometric.datasets.relbench
- torch_geometric.utils.data_warehouse
"""

import os
import sys

import torch

# Add local PyG to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import after path modification
try:
    from torch_geometric.datasets.relbench import create_relbench_hetero_data
    from torch_geometric.utils.data_warehouse import create_warehouse_demo
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure PyTorch Geometric is properly installed.")
    sys.exit(1)


def main() -> None:
    """Simple out-of-the-box warehouse intelligence demo."""
    print("Warehouse Intelligence Demo")
    print("=" * 30)

    # Step 1: Load RelBench data with warehouse labels
    print("\nStep 1: Loading RelBench data")
    try:
        hetero_data = create_relbench_hetero_data(dataset_name='rel-f1',
                                                  sample_size=50,
                                                  create_lineage_labels=True,
                                                  create_silo_labels=True,
                                                  create_anomaly_labels=True)
        print(f"Loaded graph with {len(hetero_data.node_types)} node types")
        print(f"Node types: {list(hetero_data.node_types)}")

        # Convert to homogeneous for demo
        homo_data = hetero_data.to_homogeneous()
        print(f"Converted to homogeneous: {homo_data.num_nodes} nodes, "
              f"{homo_data.num_edges} edges")

    except Exception as e:
        print(f"RelBench failed ({e}), using fallback data")
        # Create simple fallback data
        homo_data = torch.geometric.data.Data(
            x=torch.randn(50, 384), edge_index=torch.randint(0, 50, (2, 100)))

    # Step 2: Create warehouse conversation system
    print("\nStep 2: Creating warehouse conversation system")
    try:
        conversation_system = create_warehouse_demo()
        print("Warehouse conversation system ready")
    except Exception as e:
        print(f"Failed to create conversation system: {e}")
        return

    # Step 3: Prepare graph data for analysis
    graph_data = {
        'x': homo_data.x,
        'edge_index': homo_data.edge_index,
        'batch': None
    }

    # Step 3: Run warehouse intelligence queries
    print("\nStep 3: Running warehouse intelligence queries")

    queries = [
        "What is the data lineage in this warehouse?",
        "Are there any data silos?", "What is the data quality status?",
        "Analyze the impact of changes in this warehouse"
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n--- Query {i}: {query} ---")
        try:
            result = conversation_system.process_query(query, graph_data)
            print(f"Answer: {result['answer']}")
            print(f"Query type: {result['query_type']}")
            print(f"Confidence: {result.get('confidence', 'N/A')}")
        except Exception as e:
            print(f"Query failed: {e}")

    # Step 4: Show conversation history
    print("\nStep 4: Conversation History")
    print("-" * 30)
    history = conversation_system.get_conversation_history()
    for i, entry in enumerate(history[-3:], 1):  # Show last 3
        print(f"{i}. Q: {entry['query'][:50]}...")
        print(f"   A: {entry['answer'][:80]}...")

    print(f"\nDemo completed. Processed {len(history)} queries total.")
    print("\nFeatures demonstrated:")
    print("- RelBench data integration")
    print("- Multi-task warehouse intelligence")
    print("- Natural language query processing")
    print("- Lineage, silo, and quality analysis")


if __name__ == "__main__":
    main()
