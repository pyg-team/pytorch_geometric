# examples/distributed/spmd_gcn_reasoning.py
import argparse
import os
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from torch_geometric.data import Data
from torch_geometric.distributed import DistContext
from torch_geometric.distributed.edge_partition import EdgePartitioner, load_edge_partition_info
from torch_geometric.distributed.dist_edge_loader import DistEdgeLoader
from torch_geometric.nn.conv.spmd_gcn_conv import SPMDGCNConv
from torch_geometric.datasets import CoraFull


class SPMDGCN(torch.nn.Module):
    """GCN model with SPMD-style distributed message passing."""
    
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        
        # First layer
        self.layers.append(SPMDGCNConv(in_channels, hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(SPMDGCNConv(hidden_channels, hidden_channels))
        
        # Output layer
        self.layers.append(SPMDGCNConv(hidden_channels, out_channels))
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, edge_index)
            x = F.relu(x)
        return x


def train_epoch(model, data, optimizer, device):
    """Train for one epoch."""
    model.train()
    optimizer.zero_grad()
    
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    
    loss.backward()
    optimizer.step()
    
    return loss.item()


def test(model, data, device):
    """Test the model."""
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        
        train_acc = (pred[data.train_mask] == data.y[data.train_mask]).float().mean()
        val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean()
        test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()
        
        return train_acc.item(), val_acc.item(), test_acc.item()


def main():
    parser = argparse.ArgumentParser(description='SPMD GCN Training')
    parser.add_argument('--dataset', type=str, default='CoraFull', help='Dataset name')
    parser.add_argument('--num_parts', type=int, default=2, help='Number of partitions')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--hidden_channels', type=int, default=64, help='Hidden channels')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--partition_root', type=str, default='./partitions', help='Partition root')
    
    args = parser.parse_args()
    
    # Initialize distributed training
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    device = torch.device(f'cuda:{rank}')
    
    # Create distributed context
    current_ctx = DistContext(
        world_size=world_size,
        rank=rank,
        global_world_size=world_size,
        global_rank=rank,
        group_name='spmd-gcn',
    )
    
    if rank == 0:
        print("Loading and partitioning dataset...")
        # Load dataset
        dataset = CoraFull(root='./data')
        data = dataset[0]
        
        # Create edge partitions
        partitioner = EdgePartitioner(
            data=data,
            num_parts=args.num_parts,
            root=args.partition_root,
            strategy='balanced'
        )
        partitioner.generate_edge_partition()
        print("Partitioning complete!")
    
    # Wait for partitioning to complete
    dist.barrier()
    
    # Load partition data
    meta, num_parts, partition_idx, edge_partition = load_edge_partition_info(
        args.partition_root, rank
    )
    
    # Load the partition data
    partition_path = os.path.join(args.partition_root, f'part_{rank}')
    data = torch.load(os.path.join(partition_path, 'data.pt'))
    data = data.to(device)
    
    # Create model
    model = SPMDGCN(
        in_channels=data.x.size(1),
        hidden_channels=args.hidden_channels,
        out_channels=dataset.num_classes,
        num_layers=args.num_layers
    ).to(device)
    
    # Wrap with DDP
    model = DDP(model, device_ids=[rank])
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
for epoch in range(args.num_epochs):
        # Train for one epoch
        loss = train_epoch(model, data, optimizer, device)
        
        # Synchronize all processes
        dist.barrier()
        
        # Test every 10 epochs
        if epoch % 10 == 0 or epoch == args.num_epochs - 1:
            train_acc, val_acc, test_acc = test(model, data, device)
            
            # Gather results from all processes
            if rank == 0:
                print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, '
                      f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, '
                      f'Test Acc: {test_acc:.4f}')
    
    # Clean up
    dist.destroy_process_group()


if __name__ == '__main__':
    main()