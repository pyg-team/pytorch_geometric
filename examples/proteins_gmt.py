import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GraphMultisetTransformer

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'PROTEINS')
dataset = TUDataset(path, name='PROTEINS').shuffle()
average_nodes = int(dataset.data.x.size(0) / len(dataset))
n = (len(dataset) + 9) // 10
test_dataset = dataset[:n]
val_dataset = dataset[n:2 * n]
train_dataset = dataset[2 * n:]
test_loader = DataLoader(test_dataset, batch_size=128)
val_loader = DataLoader(val_dataset, batch_size=128)
train_loader = DataLoader(train_dataset, batch_size=128)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = GCNConv(dataset.num_features, 32)
        self.conv2 = GCNConv(32, 32)
        self.conv3 = GCNConv(32, 32)

        self.pool = GraphMultisetTransformer(
            96, 64, 32,
            conv=GCNConv, num_nodes=average_nodes, pooling_ratio=0.25,
            pool_sequences=['GMPool_G', 'SelfAtt', 'GMPool_I'],
            num_heads=4, ln=False
        )

        self.lin1 = torch.nn.Linear(32, 16)
        self.lin2 = torch.nn.Linear(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x1 = F.relu(self.conv1(x, edge_index))
        x2 = F.relu(self.conv2(x1, edge_index))
        x3 = F.relu(self.conv3(x2, edge_index))

        x = torch.cat([x1, x2, x3], dim=1)
        x = self.pool(x, batch, edge_index)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.log_softmax(self.lin2(x), dim=-1)

        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

print(model)


def train(epoch):
    model.train()

    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(train_dataset)


@torch.no_grad()
def test(loader):
    model.eval()
    correct = 0

    for data in loader:
        data = data.to(device)
        pred = model(data).max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


best_val_acc = test_acc = 0
for epoch in range(1, 201):
    train_loss = train(epoch)
    val_acc = test(val_loader)
    if val_acc > best_val_acc:
        test_acc = test(test_loader)
        best_val_acc = val_acc
    print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, '
          f'Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
