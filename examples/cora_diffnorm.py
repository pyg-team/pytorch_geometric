import os

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.nn import GCNConv
from torch_geometric.nn import DiffNorm


def load_cora(path='./'):
    data = Planetoid(path, name="Cora", split='public',
                     transform=T.NormalizeFeatures())[0]
    num_nodes = data.x.size(0)
    edge_index, _ = remove_self_loops(data.edge_index)
    edge_index = add_self_loops(edge_index, num_nodes=num_nodes)
    if isinstance(edge_index, tuple):
        data.edge_index = edge_index[0]
    else:
        data.edge_index = edge_index
    return data


class Net(nn.Module):
    """
    Args:
        layers: number of gcn layers.
        num_feats: number of input features.
        num_classes: number of output classes.
        gn: if True, DiffNorm will be applied after every.
            gcn layer.
    """
    def __init__(self, layers=2, num_feats=1433, num_classes=7, gn=True):
        super(Net, self).__init__()
        assert layers >= 2, 'there has to be atleast 2 layers'

        self.gn = gn
        self.num_feats = num_feats
        self.num_classes = num_classes
        self.conv = nn.ModuleList([GCNConv(self.num_feats, 16, bias=False)])
        for i in range(layers - 2):
            self.conv.append(GCNConv(16, 16, bias=False))
        self.conv.append(GCNConv(16, self.num_classes, bias=False))

        if self.gn:
            self.gnl = nn.ModuleList()
            for i in range(layers - 1):
                self.gnl.append(DiffNorm(16, 10, lamda=0.001, momentum=0.3))
            self.gnl.append(
                DiffNorm(self.num_classes, 10, lamda=0.001, momentum=0.3))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for i in range(len(self.conv)):
            x = self.conv[i](x, edge_index)
            if self.gn:
                x = self.gnl[i](x)
            x = F.relu(x)
            if i != len(self.conv) - 1:
                x = F.dropout(x, 0.6, training=self.training)

        return x


def accuracy(y, y_hat):
    return float(torch.sum(y == y_hat)) / len(y)


def trainer(data, epochs=1000, layers=3, model_path='cora_models',
            diff_norm=True, device='cuda') -> (Net, float, float):
    '''
    Trains a simple gcn on cora dataset. Model is saved every.
    time validation accuracy goes down.
    Returns model, and groupdistanceratio for train and test sets.
    Args:
        layers: number of gcn layers.
        model_path: location to save models.
        data:Cora dataset.
        diff_norm: if True differentiable group norm will be applied after
            conv layer.

    '''
    assert layers >= 2
    distratio = DiffNorm.groupDistanceRatio
    gd = 0  # group distance for test set
    gd_train = 0  # group distance for train set

    model = Net(layers=layers, gn=diff_norm).to(device=device)
    model_path = os.path.join(model_path, f'cora_{layers}.pt')

    data = data.to(device=device)
    loss = nn.CrossEntropyLoss()
    optim = Adam(model.parameters(), lr=0.005, weight_decay=0.0005)

    max_accu = 0  # best validation accuracy
    best_epoch = 0  # epoch of best val accuracy

    for i in range(epochs):
        model.train()
        out = model(data)[data.train_mask]
        loss(out, data.y[data.train_mask]).backward()
        optim.step()
        optim.zero_grad()

        with torch.no_grad():
            model.eval()
            out = model.forward(data)[data.val_mask]
            accu = accuracy(out.argmax(dim=1), data.y[data.val_mask])
            if accu > max_accu:
                torch.save(model.state_dict(), model_path)
                best_epoch = i
                max_accu = accu

            if i % 100 == 0:
                print(f'Epoch:{i} Validation accuracy={accu}')

    with torch.no_grad():
        print(f'Best val accuracy at epoch: {best_epoch}.')
        model.load_state_dict(torch.load(model_path))
        out = model(data)[data.test_mask]
        print(f'Test Acc={accuracy(out.argmax(dim=1),data.y[data.test_mask])}')
        gd = distratio(out, data.y[data.test_mask].to(device=device))

        model.load_state_dict(torch.load(model_path))
        out = model(data)[data.train_mask]
        print(
            f'Train Acc={accuracy(out.argmax(dim=1),data.y[data.train_mask])}')
        gd_train = distratio(out, data.y[data.train_mask].to(device=device))
        return model, gd, gd_train


data = load_cora()
trainer(data, layers=2,
        diff_norm=True)  # with 2 layers test accuracy is around 81%.
