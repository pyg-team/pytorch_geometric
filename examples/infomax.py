import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.nn.inits import uniform

hidden_dim = 512

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
data = Planetoid(path, dataset)[0]


class Encoder(nn.Module):
    def __init__(self, hidden_dim):
        super(Encoder, self).__init__()
        self.conv = GCNConv(data.num_features, hidden_dim)
        self.prelu = nn.PReLU(hidden_dim)

    def forward(self, x, edge_index, corrupt=False):
        if corrupt:
            perm = torch.randperm(data.num_nodes)
            x = x[perm]

        x = self.conv(x, edge_index)
        x = self.prelu(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, hidden_dim):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        size = self.weight.size(0)
        uniform(size, self.weight)

    def forward(self, x, summary):
        x = torch.matmul(x, torch.matmul(self.weight, summary))
        return x


class Infomax(nn.Module):
    def __init__(self, hidden_dim):
        super(Infomax, self).__init__()
        self.encoder = Encoder(hidden_dim)
        self.discriminator = Discriminator(hidden_dim)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x, edge_index):
        positive = self.encoder(x, edge_index, corrupt=False)
        negative = self.encoder(x, edge_index, corrupt=True)
        summary = torch.sigmoid(positive.mean(dim=0))

        positive = self.discriminator(positive, summary)
        negative = self.discriminator(negative, summary)

        l1 = self.loss(positive, torch.ones_like(positive))
        l2 = self.loss(negative, torch.zeros_like(negative))

        return l1 + l2


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)
infomax = Infomax(hidden_dim).to(device)
infomax_optimizer = torch.optim.Adam(infomax.parameters(), lr=0.001)


def train_infomax(epoch):
    infomax.train()

    if epoch == 200:
        for param_group in infomax_optimizer.param_groups:
            param_group['lr'] = 0.0001

    infomax_optimizer.zero_grad()
    loss = infomax(data.x, data.edge_index)
    loss.backward()
    infomax_optimizer.step()
    return loss.item()


print('Train deep graph infomax.')
for epoch in range(1, 301):
    loss = train_infomax(epoch)
    print('Epoch: {:03d}, Loss: {:.7f}'.format(epoch, loss))


class Classifier(nn.Module):
    def __init__(self, hidden_dim):
        super(Classifier, self).__init__()
        self.lin = nn.Linear(hidden_dim, data.num_classes)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, x, edge_index):
        x = infomax.encoder(x, edge_index, corrupt=False)
        x = x.detach()
        x = self.lin(x)
        return torch.log_softmax(x, dim=-1)


classifier = Classifier(hidden_dim).to(device)
classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)


def train_classifier():
    infomax.eval()
    classifier.train()
    classifier_optimizer.zero_grad()
    output = classifier(data.x, data.edge_index)
    loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
    loss.backward()
    classifier_optimizer.step()
    return loss.item()


def test_classifier():
    infomax.eval()
    classifier.eval()
    logits, accs = classifier(data.x, data.edge_index), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


print('Train logistic regression classifier.')
for epoch in range(1, 51):
    train_classifier()
    accs = test_classifier()
    log = 'Epoch: {:02d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, *accs))
