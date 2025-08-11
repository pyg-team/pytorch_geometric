import os.path as osp

import torch
from torch.nn import CrossEntropyLoss

from torch_geometric.datasets import Planetoid
from torch_geometric.nn.models.gkan import GKANModel

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the Cora dataset
path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data", "Cora")
dataset = Planetoid(root=path, name="Cora")
data = dataset[0].to(device)

# Model configuration
input_dim = dataset.num_features
hidden_dim = 64
output_dim = dataset.num_classes
num = 5  # Number of grid intervals for KANLayer
k = 3  # Spline polynomial order
num_layers = 3
architecture = 2  # 1 for Aggregate->KAN, 2 for KAN->Aggregate

# Initialize the GKANModel
model = GKANModel(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    num_layers=num_layers,
    num=num,  # Grid intervals for KANLayer
    k=k,  # Spline polynomial order
    architecture=architecture,
).to(device)

# Optimizer and Loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = CrossEntropyLoss()


def train():
    """Train the GKAN model."""
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)  # Forward pass
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test():
    """Evaluate the GKAN model."""
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    acc = (pred[data.test_mask] == data.y[data.test_mask]
           ).sum().item() / data.test_mask.sum().item()
    return acc


# Training Loop
print("Training GKANModel on Cora...")
for epoch in range(1, 201):  # 200 epochs
    loss = train()
    print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")

# Testing
accuracy = test()
print(f"\nTest Accuracy: {accuracy:.4f}")
