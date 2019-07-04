import os.path as osp
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import SyntheticDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GraphConv, GINConv, TopKPooling
from torch_geometric.nn import global_add_pool as gsum
from torch_scatter import scatter_mean


train_path = osp.join(osp.dirname(osp.realpath(__file__)),
                      '..', 'data', 'COLORS3_train')
train_dataset = SyntheticDataset(train_path,
                                 name='COLORS3_train', use_node_attr=True)
train_dataset = train_dataset.shuffle()
train_loader = DataLoader(train_dataset, batch_size=60)


val_path = osp.join(osp.dirname(osp.realpath(__file__)),
                    '..', 'data', 'COLORS3_val')
val_dataset = SyntheticDataset(val_path,
                               name='COLORS3_val', use_node_attr=True)
val_loader = DataLoader(val_dataset, batch_size=60)


test_path = osp.join(osp.dirname(osp.realpath(__file__)),
                     '..', 'data', 'COLORS3_test')
test_dataset = SyntheticDataset(test_path,
                                name='COLORS3_test', use_node_attr=True)
test_loader = DataLoader(test_dataset, batch_size=60)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = GINConv(nn.Sequential(nn.Linear(train_dataset.
                                                     num_features, 256),
                                           nn.ReLU(),
                                           nn.Linear(256, 64)))
        self.pool1 = TopKPooling(train_dataset.num_features, min_score=0.05)
        self.conv2 = GINConv(nn.Sequential(nn.Linear(64, 256),
                                           nn.ReLU(),
                                           nn.Linear(256, 64)))

        self.lin = torch.nn.Linear(64, 1)  # regression

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x_input = x
        x = F.relu(self.conv1(x_input, edge_index))

        x, edge_index, _, batch, perm, score = self.pool1(x, edge_index,
                                                          None, batch,
                                                          attn_input=x_input)
        ratio = x.shape[0] / float(x_input.shape[0])

        x = F.relu(self.conv2(x, edge_index))
        x = gsum(x, batch)
        x = self.lin(x)

        # supervised node attention
        attn_loss_batch = scatter_mean(F.kl_div(torch.log(score + 1e-14),
                                                data.node_attention[perm],
                                                reduction='none'), batch)

        return x, attn_loss_batch, ratio


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
# Initialize to optimal attention weights:
model.pool1.weight.data = torch.tensor([0., 1., 0., 0.]).view(1, 4).to(device)

print(model)
print('model size: %d trainable parameters' %
      np.sum([np.prod(p.size()) if p.requires_grad else 0
              for p in model.parameters()]))


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train(epoch):
    model.train()

    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output, attn_loss, _ = model(data)
        loss = ((data.y - output.view_as(data.y)) ** 2 + 100*attn_loss).mean()

        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()

    return loss_all / len(train_dataset)


def test(loader):
    model.eval()

    correct, ratio_all = 0, 0
    for data in loader:
        data = data.to(device)
        output, _, ratio = model(data)
        pred = output.round().long().view_as(data.y)
        correct += pred.eq(data.y.long()).sum().item()
        ratio_all += ratio
    return correct, correct / len(loader.dataset), ratio_all / len(loader)


for epoch in range(1, 301):
    loss = train(epoch)
    train_correct, train_acc, train_ratio = test(train_loader)
    val_correct, val_acc, val_ratio = test(val_loader)
    test_correct, test_acc, test_ratio = test(test_loader)
    print('Epoch: {:03d}, Loss: {:.5f}, Train Acc: {:.5f}, Val Acc: {:.5f}, '
          'Test Acc: {:.5f} ({}/{}), '
          'Train/Val/Test Pool Ratio={:.3f}/{:.3f}/{:.3f}'.
          format(epoch, loss, train_acc, val_acc,
                 test_acc, test_correct, len(test_loader.dataset),
                 train_ratio, val_ratio, test_ratio))
