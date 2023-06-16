import os.path as osp

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import NBFNet
from torch_geometric.sampler import NegativeSampling

torch.manual_seed(123)

num_negatives = 1
batch_size = 64 * num_negatives
epochs = 20
lr = 5.0e-3
num_layers = 6
hidden_channels = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = T.Compose([
    T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                      add_negative_train_samples=False,
                      neg_sampling_ratio=num_negatives)
])
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
dataset = Planetoid(path, name='Cora', transform=transform)
# After applying the `RandomLinkSplit` transform, the data is transformed from
# a data object to a list of tuples (train_data, val_data, test_data), with
# each element representing the corresponding split.
train_data, val_data, test_data = dataset[0]

train_loader = LinkNeighborLoader(
    train_data, num_neighbors=[-1 for _ in range(train_data.num_nodes)],
    edge_label_index=train_data.edge_label_index,
    neg_sampling=NegativeSampling(mode="binary", amount=num_negatives),
    batch_size=batch_size, shuffle=True)

val_loader = LinkNeighborLoader(
    val_data, num_neighbors=[-1 for _ in range(val_data.num_nodes)],
    edge_label_index=val_data.edge_label_index, edge_label=val_data.edge_label,
    batch_size=batch_size, shuffle=True)

test_loader = LinkNeighborLoader(
    test_data, num_neighbors=[-1 for _ in range(test_data.num_nodes)],
    edge_label_index=test_data.edge_label_index,
    edge_label=test_data.edge_label, batch_size=batch_size, shuffle=True)

model = NBFNet(dataset.num_features, num_layers, hidden_channels).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = torch.nn.BCELoss(reduction='none')


def train(train_loader, model, optimizer, criterion):
    model.train()
    aucs, aps, losses = [], [], []
    for batch in train_loader:  # there are less nodes and edges than expected?
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch.edge_label_index, batch.edge_index).view(-1)
        target = batch.edge_label
        loss = criterion(pred, target)
        weight = torch.ones_like(pred)
        weight[target == 0] = 1 / num_negatives
        loss = (loss * weight).sum() / weight.sum()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        auc = roc_auc_score(target.cpu().detach().numpy(),
                            pred.cpu().detach().numpy())
        aucs.append(auc)
        ap = average_precision_score(target.cpu().detach().numpy(),
                                     pred.cpu().detach().numpy())
        aps.append(ap)
    return np.mean(losses), np.mean(aucs), np.mean(aps)


@torch.no_grad()
def test(test_loader, model, criterion):
    model.eval()
    aucs, aps, losses = [], [], []
    for batch in test_loader:
        batch = batch.to(device)
        pred = model(batch.edge_label_index, batch.edge_index).view(-1)
        target = batch.edge_label
        loss = criterion(pred, target)
        neg_weight = torch.ones_like(pred)
        neg_weight[target == 0] = 1 / num_negatives
        loss = (loss * neg_weight).sum() / neg_weight.sum()
        losses.append(loss.item())
        auc = roc_auc_score(target.cpu().detach().numpy(),
                            pred.cpu().detach().numpy())
        aucs.append(auc)
        ap = average_precision_score(target.cpu().detach().numpy(),
                                     pred.cpu().detach().numpy())
        aps.append(ap)
    return np.mean(losses), np.mean(aucs), np.mean(aps)


train_losses = []
train_aucs = []
train_aps = []
val_losses = []
val_aucs = []
val_aps = []

for epoch in range(epochs):
    train_loss, train_auc, train_ap = train(train_loader, model, optimizer,
                                            criterion)
    train_losses.append(train_loss)
    train_aucs.append(train_auc)
    train_aps.append(train_ap)
    val_loss, val_auc, val_ap = test(val_loader, model, criterion)
    val_losses.append(val_loss)
    val_aucs.append(val_auc)
    val_aps.append(val_ap)
    print(f"Epoch: {epoch},\
    Train Loss: {train_loss}, Train AUC: {train_auc}, Train AP: {train_ap},\
    Val Loss: {val_loss}, Val AUC: {val_auc}, Val AP: {val_ap}")

print("Training finished!")
test_loss, test_auc, test_ap = test(test_loader, model, criterion)
print(f"Test Loss: {test_loss}, Test AUC: {test_auc}, Test AP: {test_ap}")
