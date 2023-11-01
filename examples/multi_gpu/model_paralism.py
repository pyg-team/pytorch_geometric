import torch
import torch.nn.functional as F
from torch.nn import Linear

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

dataset = Planetoid(root='data/Planetoid', name='Cora',
                    transform=NormalizeFeatures())

print()
print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

data = dataset[0]  # Get the first graph object.

print()
print(data)
print(
    '==========================================================================================================='
)

# Gather some statistics about the graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Number of training nodes: {data.train_mask.sum()}')
print(
    f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}'
)
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')

# model on multiple GPUs


class MLP(torch.nn.Module):
    def __init__(self, hidden_channels, device1="cuda:0", device2="cuda:1"):
        super().__init__()
        torch.manual_seed(12345)
        self.device1 = device1
        self.device2 = device2
        self.lin1 = Linear(dataset.num_features,
                           hidden_channels).to(self.device1)
        self.lin2 = Linear(hidden_channels,
                           dataset.num_classes).to(self.device2)

    def forward(self, x):
        x = x.to(self.device1)
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.to(self.device2)
        x = self.lin2(x)
        return x


model = MLP(hidden_channels=16)
criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
optimizer = torch.optim.Adam(model.parameters(), lr=0.01,
                             weight_decay=5e-4)  # Define optimizer.


def train():
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = model(data.x)  # Perform a single forward pass.
    data = data.to("cuda:1")
    loss = criterion(out[data.train_mask], data.y[data.train_mask]
                     )  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss


def test():
    model.eval()
    out = model(data.x)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    data = data.to("cuda:1")
    test_correct = pred[data.test_mask] == data.y[
        data.test_mask]  # Check against ground-truth labels.
    test_acc = int(test_correct.sum()) / int(
        data.test_mask.sum())  # Derive ratio of correct predictions.
    return test_acc


for epoch in range(1, 201):
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
