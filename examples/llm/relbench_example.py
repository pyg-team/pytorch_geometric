"""RelBench integration example for PyTorch Geometric."""

import argparse

from torch_geometric.utils.relbench import create_relbench_hetero_data


def main():
    """Basic RelBench integration example."""
    parser = argparse.ArgumentParser(description='RelBench Example')
    parser.add_argument('--dataset', type=str, default='rel-trial',
                        help='RelBench dataset name')
    parser.add_argument('--sample-size', type=int, default=50,
                        help='Sample size per table')
    args = parser.parse_args()

    print("RelBench GNN+LLM Integration Example")
    print("-" * 40)

    try:
        print(f"Loading RelBench dataset: {args.dataset}")
        hetero_data = create_relbench_hetero_data(args.dataset,
                                                  sample_size=args.sample_size)
        print("Dataset loaded successfully")

        print("HeteroData Summary:")
        print(f"  Node types: {list(hetero_data.node_types)}")
        for node_type in hetero_data.node_types:
            if hasattr(hetero_data[node_type], 'x'):
                x = hetero_data[node_type].x
                print(f"  {node_type}: {x.shape[0]} nodes, "
                      f"{x.shape[1]} features")

        print("RelBench integration completed successfully")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
