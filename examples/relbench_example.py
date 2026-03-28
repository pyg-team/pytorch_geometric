"""Example demonstrating how to use ``from_relbench`` to convert a RelBench
relational database into a PyG HeteroData graph and train a heterogeneous
GNN for node-level prediction.

This example loads the Formula 1 RelBench dataset, converts it into a
heterogeneous graph using ``from_relbench``, and trains a 2-layer GraphSAGE
model (via ``to_hetero``) to predict championship standings points from
the graph structure and node features.

Requirements:
    ``pip install relbench``

Usage:
    ``python relbench_example.py``
    ``python relbench_example.py --epochs 50 --hidden_channels 128``
"""

import argparse
from typing import Tuple

import torch
import torch.nn.functional as F
from relbench.datasets import get_dataset

from torch_geometric.nn import Linear, SAGEConv, to_hetero
from torch_geometric.utils import from_relbench

parser = argparse.ArgumentParser(
    description='Train a heterogeneous GNN on a RelBench dataset.')
parser.add_argument('--hidden_channels', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--epochs', type=int, default=30)
args = parser.parse_args()

torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Load a RelBench dataset and convert to HeteroData:
print('Loading RelBench rel-f1 dataset...')
dataset = get_dataset('rel-f1', download=True)
db = dataset.get_db()
data = from_relbench(db)
print(f'Graph: {len(data.node_types)} node types, '
      f'{len(data.edge_types)} edge types')

# 2. Prepare a node regression target.
# `from_relbench` preserves the original DataFrame column order from RelBench.
# In rel-f1, the 'standings' table has 'points' as its first numeric column:
target_type = 'standings'
y = data[target_type].x[:, 0]  # points column (index 0 in rel-f1)
data[target_type].x = data[target_type].x[:, 1:]  # remove from input features

# 3. Clean up features — fill NaN and standardize per column:
for node_type in data.node_types:
    if hasattr(data[node_type], 'x') and data[node_type].x is not None:
        x = torch.nan_to_num(data[node_type].x, nan=0.0)
        std, mean = torch.std_mean(x, dim=0)
        std[std == 0] = 1.0  # avoid division by zero for constant columns
        data[node_type].x = (x - mean) / std
    else:
        # Zero-feature placeholder for featureless node types (e.g. drivers):
        data[node_type].x = torch.zeros(data[node_type].num_nodes, 1)

# 4. Create train/val/test splits (60/20/20) before computing target stats:
num_nodes = data[target_type].num_nodes
perm = torch.randperm(num_nodes)
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
val_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)
train_mask[perm[:int(0.6 * num_nodes)]] = True
val_mask[perm[int(0.6 * num_nodes):int(0.8 * num_nodes)]] = True
test_mask[perm[int(0.8 * num_nodes):]] = True

# Normalize target using training set statistics only (prevents data leakage):
y_mean = y[train_mask].mean()
y_std = y[train_mask].std()
y_std = y_std if y_std > 0 else torch.tensor(1.0)
y_norm = (y - y_mean) / y_std

# 5. Move all tensors to device — including masks to prevent device mismatch:
data = data.to(device)
y = y.to(device)
y_norm = y_norm.to(device)
train_mask = train_mask.to(device)
val_mask = val_mask.to(device)
test_mask = test_mask.to(device)


# 6. Define a 2-layer GraphSAGE model with lazy input size inference:
class GNN(torch.nn.Module):
    def __init__(self, hidden_channels: int) -> None:
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels)
        self.lin = Linear(-1, 1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return self.lin(x)


model = GNN(args.hidden_channels)
model = to_hetero(model, data.metadata(), aggr='sum').to(device)

# Initialize lazy parameters via a single dry-run forward pass:
with torch.no_grad():
    model(data.x_dict, data.edge_index_dict)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


def train() -> torch.Tensor:
    model.train()
    optimizer.zero_grad()
    pred = model(data.x_dict, data.edge_index_dict)[target_type].squeeze(-1)
    loss = F.mse_loss(pred[train_mask], y_norm[train_mask])
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def test() -> Tuple[float, float, float]:
    model.eval()
    pred = model(data.x_dict, data.edge_index_dict)[target_type].squeeze(-1)
    # denormalize for interpretable MAE
    pred *= y_std
    pred += y_mean

    train_mae = float((pred[train_mask] - y[train_mask]).abs().mean())
    val_mae = float((pred[val_mask] - y[val_mask]).abs().mean())
    test_mae = float((pred[test_mask] - y[test_mask]).abs().mean())
    return train_mae, val_mae, test_mae


print(
    f'\nTraining {args.epochs} epochs on "{target_type}" point prediction...')
print(f'Target stats (train): mean={y_mean:.2f}, std={y_std:.2f}\n')

for epoch in range(1, args.epochs + 1):
    loss = train()
    if epoch % 5 == 0 or epoch == 1:
        train_mae, val_mae, test_mae = test()
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
              f'Train MAE: {train_mae:.2f}, Val MAE: {val_mae:.2f}, '
              f'Test MAE: {test_mae:.2f} points')

train_mae, val_mae, test_mae = test()
print(f'\nFinal — Train MAE: {train_mae:.2f}, Val MAE: {val_mae:.2f}, '
      f'Test MAE: {test_mae:.2f} points')
