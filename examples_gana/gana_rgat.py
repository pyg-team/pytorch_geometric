import argparse
import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Entities
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.nn import RGATConv
from torch_geometric.loader import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('-k', '--num_layers', default=2, help="number of layers")
parser.add_argument('-epochs', '--epochs', default=2, help="number of epochs")
parser.add_argument('-hidden_channels', '--hidden_channels', default=16, help="number of channels in hidden layers")
args = parser.parse_args()

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Entities')

train_dataset = Entities(path, 'ppi', split='train')
val_dataset = Entities(path, 'ppi', split='val')
test_dataset = Entities(path, 'ppi', split='test')
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

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
            RGATConv(train_dataset.num_features, hidden_channels,
                     num_relations))
        for _ in range(num_layers - 2):
            self.convs.append(
                RGATConv(hidden_channels, hidden_channels, num_relations))
        self.convs.append(RGATConv(hidden_channels, num_classes, num_relations))
        print(
            f"number of layers k:{len(self.convs)} hidden layersize :{hidden_channels}")

    def forward(self, data):
        x, edge_index, edge_type = data.x, data.edge_index, data.edge_type
        for conv in self.convs:
            x1 = conv(x, edge_index, edge_type)

            x = F.relu(x1)
            x = F.dropout(x, training=self.training)

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
model = Net(int(args.num_layers), int(args.hidden_channels)).to(device)
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


for epoch in range(1, int(args.epochs)):
    loss = train(train_loader)
    # train_acc = test(train_loader)
    val_f1 = test(val_loader)
    test_f1 = test(test_loader)
    print('Epoch: {:02d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}'.format(
        epoch, loss, val_f1, test_f1))
