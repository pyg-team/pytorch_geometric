"""Example: Using HeteroConv with TensorBoard Visualization
==========================================================

This example demonstrates how to use the `jit_trace_friendly` wrapper
to visualize HeteroConv models with TensorBoard, resolving the issue
where torch.jit.trace doesn't support tuple dictionary keys.

Issue: https://github.com/pyg-team/pytorch_geometric/issues/10421
"""

import torch
from torch.utils.tensorboard import SummaryWriter

import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
from torch_geometric.nn import GATConv, GCNConv, HeteroConv, Linear, SAGEConv


# Define a heterogeneous GNN model
class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    ('paper', 'cites', 'paper'):
                    GCNConv(-1, hidden_channels),
                    ('author', 'writes', 'paper'):
                    SAGEConv((-1, -1), hidden_channels),
                    ('paper', 'rev_writes', 'author'):
                    GATConv((-1, -1), hidden_channels, add_self_loops=False),
                }, aggr='sum')
            self.convs.append(conv)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        return self.lin(x_dict['author'])


def main():
    # Load a heterogeneous graph dataset
    print("Loading dataset...")
    dataset = OGB_MAG(root='./data', preprocess='metapath2vec',
                      transform=T.ToUndirected())
    data = dataset[0]

    # Instantiate the model and initialize lazy modules
    print("Initializing model...")
    model = HeteroGNN(hidden_channels=64, out_channels=dataset.num_classes,
                      num_layers=2)
    with torch.no_grad():
        _ = model(data.x_dict, data.edge_index_dict)

    # Prepare list-based inputs for the wrapper
    x_list = list(data.x_dict.values())
    x_dict_keys = list(data.x_dict.keys())
    edge_index_list = list(data.edge_index_dict.values())
    edge_index_dict_keys = list(data.edge_index_dict.keys())

    print(f"\nNode types: {x_dict_keys}")
    print(f"Number of edge types: {len(edge_index_dict_keys)}")

    # Create a JIT-friendly wrapper for a single HeteroConv layer
    print("\nCreating JIT-friendly wrapper...")
    single_conv = model.convs[0]
    wrapped_conv = single_conv.jit_trace_friendly(x_dict_keys,
                                                  edge_index_dict_keys)

    # Test forward pass
    print("Testing forward pass...")
    with torch.no_grad():
        out_list = wrapped_conv(x_list, edge_index_list)
        print(f"✅ Forward pass successful!")
        print(f"   Output list length: {len(out_list)}")
        print(f"   Output shapes: {[o.shape for o in out_list]}")

    # Visualize with TensorBoard
    print("\nAdding graph to TensorBoard...")
    writer = SummaryWriter('runs/hetero_conv_example')
    try:
        writer.add_graph(wrapped_conv, (x_list, edge_index_list))
        print("✅ TensorBoard visualization successful!")
        print("   Run 'tensorboard --logdir=runs' to view the graph")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        writer.close()

    # Test torch.jit.trace compatibility
    print("\nTesting torch.jit.trace compatibility...")
    try:
        traced_model = torch.jit.trace(wrapped_conv, (x_list, edge_index_list))
        print("✅ torch.jit.trace successful!")

        # Verify traced model works
        with torch.no_grad():
            traced_model(x_list, edge_index_list)
            print(f"✅ Traced model forward pass successful!")
    except Exception as e:
        print(f"❌ Error: {e}")

    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
