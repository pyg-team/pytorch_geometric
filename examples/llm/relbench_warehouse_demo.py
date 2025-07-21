"""RelBench Data Warehouse Integration Demo for PyTorch Geometric.

This example demonstrates RelBench dataset conversion to PyG HeteroData format
for data warehouse applications including lineage tracking, silo detection,
and anomaly identification.

Complements examples/rdl.py by providing warehouse-specific utilities
and G-Retriever preparation for future LLM integration.

Requirements:
`pip install relbench[full] sentence-transformers`

Paper references:
- RelBench: https://arxiv.org/abs/2407.20060
- G-Retriever: https://arxiv.org/abs/2402.07630
"""
import argparse
import time
from typing import Any, Dict, Tuple

import torch

from torch_geometric import seed_everything
from torch_geometric.data import HeteroData
from torch_geometric.datasets.relbench import (
    RelBenchProcessor,
    create_relbench_hetero_data,
    get_warehouse_task_info,
    prepare_for_gretriever,
)


def demonstrate_basic_usage(dataset_name: str,
                            sample_size: int = 100) -> HeteroData:
    """Demonstrate basic RelBench to PyG conversion.

    Args:
        dataset_name: RelBench dataset name (e.g., 'rel-trial')
        sample_size: Number of records to sample

    Returns:
        HeteroData object with warehouse enhancements
    """
    print(f"Converting RelBench dataset '{dataset_name}' to PyG format...")
    start_time = time.time()

    hetero_data = create_relbench_hetero_data(dataset_name=dataset_name,
                                              sample_size=sample_size,
                                              add_warehouse_labels=True)

    conversion_time = time.time() - start_time
    print(f"Conversion completed in {conversion_time:.2f}s")

    # Display basic statistics
    print("Graph Statistics:")
    print(f"   Node types: {list(hetero_data.node_types)}")
    print(f"   Edge types: {list(hetero_data.edge_types)}")
    total_nodes = sum(hetero_data[node_type].num_nodes
                      for node_type in hetero_data.node_types)
    total_edges = sum(hetero_data[edge_type].num_edges
                      for edge_type in hetero_data.edge_types)
    print(f"   Total nodes: {total_nodes}")
    print(f"   Total edges: {total_edges}")

    return hetero_data


def demonstrate_warehouse_tasks() -> Dict[str, Any]:
    """Demonstrate warehouse-specific task definitions.

    Returns:
        Dictionary containing warehouse task information
    """
    print("\nWarehouse Task Information:")

    task_info = get_warehouse_task_info()

    for task_name, task_data in task_info.items():
        print(f"   {task_name.upper()}:")
        classes_str = ', '.join(task_data['classes'])
        print(f"      Classes: {task_data['num_classes']} ({classes_str})")
        print(f"      Description: {task_data['description']}")

    return task_info


def demonstrate_processor_usage(
        sbert_model: str = 'all-MiniLM-L6-v2') -> RelBenchProcessor:
    """Demonstrate RelBenchProcessor with custom SBERT model.

    Args:
        sbert_model: SBERT model name for embeddings

    Returns:
        Configured RelBenchProcessor instance
    """
    print("\nRelBench Processor Configuration:")

    processor = RelBenchProcessor(sbert_model=sbert_model)
    print(f"   Model: {processor.sbert_model_name}")
    print("   Embedding dimension: 384 (SBERT)")
    print("   Optimized for: Semantic similarity and Q&A tasks")

    return processor


def demonstrate_gretriever_preparation(
        hetero_data: HeteroData) -> Tuple[HeteroData, Dict[str, Any]]:
    """Demonstrate G-Retriever preparation for future LLM integration.

    Args:
        hetero_data: Input HeteroData object

    Returns:
        Tuple of (enhanced_hetero_data, metadata_dict)
    """
    print("\nG-Retriever Preparation:")

    enhanced_data, metadata = prepare_for_gretriever(hetero_data)

    print(f"   G-Retriever ready: {enhanced_data.gretriever_ready}")
    print(f"   Embedding type: {enhanced_data.embedding_type}")
    print(f"   Warehouse enhanced: {enhanced_data.warehouse_enhanced}")

    print(f"   Metadata keys: {list(metadata.keys())}")
    print(f"   Warehouse tasks: {metadata['warehouse_tasks']}")
    print(f"   Conversion ready: {metadata['conversion_ready']}")

    return enhanced_data, metadata


def save_demo_results(hetero_data: HeteroData, metadata: Dict[str, Any],
                      save_path: str = 'relbench_demo_output.pt'):
    """Save demonstration results for future use.

    Args:
        hetero_data: Enhanced HeteroData object
        metadata: G-Retriever metadata
        save_path: Path to save results
    """
    print("\nSaving Results:")

    results = {
        'hetero_data': hetero_data,
        'metadata': metadata,
        'timestamp': time.time()
    }

    torch.save(results, save_path)
    print(f"   Saved to: {save_path}")


def main(dataset_name: str, sample_size: int, sbert_model: str,
         save_results: bool, seed: int):
    """Main demonstration function.

    Args:
        dataset_name: RelBench dataset name
        sample_size: Number of records to sample
        sbert_model: SBERT model for embeddings
        save_results: Whether to save results
        seed: Random seed for reproducibility
    """
    seed_everything(seed)

    print("RelBench Data Warehouse Integration Demo")
    print("=" * 60)

    try:
        hetero_data = demonstrate_basic_usage(dataset_name, sample_size)
        demonstrate_warehouse_tasks()
        demonstrate_processor_usage(sbert_model)
        enhanced_data, metadata = demonstrate_gretriever_preparation(
            hetero_data)

        if save_results:
            save_demo_results(enhanced_data, metadata)

        print("\nDemo completed successfully!")
        print("Ready for G-Retriever integration and warehouse Q&A "
              "applications")

    except ImportError as e:
        if 'relbench' in str(e).lower():
            print(f"RelBench not available: {e}")
            print("Install with: pip install relbench[full] "
                  "sentence-transformers")
        else:
            raise e
    except Exception as e:
        print(f"Demo failed: {e}")
        raise e


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='RelBench Data Warehouse Integration Demo')
    parser.add_argument('--dataset', type=str, default='rel-trial',
                        help='RelBench dataset name (default: rel-trial)')
    parser.add_argument('--sample_size', type=int, default=100,
                        help='Number of records to sample (default: 100)')
    parser.add_argument(
        '--sbert_model', type=str, default='all-MiniLM-L6-v2',
        help='SBERT model for embeddings '
        '(default: all-MiniLM-L6-v2)')
    parser.add_argument('--save_results', action='store_true',
                        help='Save demonstration results to file')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()

    start_time = time.time()
    main(args.dataset, args.sample_size, args.sbert_model, args.save_results,
         args.seed)
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f}s")
