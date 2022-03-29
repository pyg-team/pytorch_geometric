import argparse
import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Entities
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.nn import RGCNConv, FastRGCNConv
from torch_geometric.loader import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,
                    choices=['AIFB', 'MUTAG', 'BGS', 'AM'])
args = parser.parse_args()

# Trade memory consumption for faster computation.
if args.dataset in ['AIFB', 'MUTAG']:
    RGCNConv = FastRGCNConv

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Entities')

train_dataset = Entities(path, 'ppi', split='train')
val_dataset = Entities(path, 'ppi', split='val')
test_dataset = Entities(path, 'ppi', split='test')
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

print(train_dataset[0])
num_relations = train_dataset.num_relations
num_classes = train_dataset.num_classes
num_features = train_dataset.num_features
print(num_relations, num_classes, num_features)


class Net(torch.nn.Module):
    def __init__(self, num_layers, hidden_channels):
        super(Net, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            RGCNConv(train_dataset.num_features, hidden_channels,
                     num_relations, num_bases=30))
        for _ in range(num_layers - 2):
            self.convs.append(
                RGCNConv(hidden_channels, hidden_channels, num_relations,
                         num_bases=30))
        self.convs.append(RGCNConv(hidden_channels, num_classes, num_relations,
                                   num_bases=30))
        print(
            f"number of layers k:{len(self.convs)} hidden layersize :{hidden_channels}")

    def forward(self, data):
        x, edge_index, edge_type = data.x, data.edge_index, data.edge_type
        for conv in self.convs:
            x1 = conv(x, edge_index, edge_type)
            
            x = F.relu(x1)
        return F.log_softmax(x1, dim=1)


# class Net(torch.nn.Module):
#     def __init__(self, num_layers, hidden_channels):
#         super().__init__()
#         self.conv1 = RGCNConv(train_dataset.num_features, 16, num_relations,
#                               num_bases=30)
#         self.conv2 = RGCNConv(16, num_classes, num_relations,
#                               num_bases=30)

#     def forward(self, data):
#         x, edge_index, edge_type = data.x, data.edge_index, data.edge_type
#         x = F.relu(self.conv1(x, edge_index, edge_type))
#         x = self.conv2(x, edge_index, edge_type)
#         return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu') if args.dataset == 'AM' else device
model = Net(4, 16).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)


def train(train_loader):
    model.train()

    total_loss = total_examples = 0
    # print(f"{train_loader}")
    for data in train_loader:
        data = data.to(device)
        # print(f"data {data}")
        optimizer.zero_grad()
        # print(data.edge_index, data.edge_type)
        # print(f"batch size {data.x.size(0)}")
        loss = F.nll_loss(model(data), data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_nodes
        total_examples += data.num_nodes
    return total_loss / total_examples


@torch.no_grad()
def test(loader):
    model.eval()

    ys, preds = [], []
    for data in loader:
        ys.append(data.y)
        data = data.to(device)
        pred = model(data).argmax(dim=-1)
        preds.append(pred.cpu())

    y = torch.cat(ys, dim=0).to(torch.float)
    pred = torch.cat(preds, dim=0).to(torch.float)
    acc = pred.eq(y).to(torch.float).mean()
    return acc.item()


for epoch in range(1, 41):
    loss = train(train_loader)
    train_acc = test(train_loader)
    val_acc = test(val_loader)
    test_acc = test(test_loader)
    print(
        f'Epoch: {epoch:02d}, Loss: {loss:.4f}, \
            Train: {train_acc:.4f}, Val accuracy: {val_acc:.4f}, Test accuracy: {test_acc:.4f}')
