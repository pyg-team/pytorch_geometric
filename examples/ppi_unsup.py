import os.path as osp

import torch
import torch.nn.functional as F
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.multioutput import MultiOutputClassifier

from torch_geometric.data import Batch
from torch_geometric.datasets import PPI
from torch_geometric.loader import DataLoader, LinkNeighborLoader
from torch_geometric.nn import GraphSAGE

EPS = 1e-15

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'PPI')
train_dataset = PPI(path, split='train')
val_dataset = PPI(path, split='val')
test_dataset = PPI(path, split='test')

# Training loaders (one datapoint corresponds to a single node)
train_data = Batch.from_data_list(train_dataset)
loader = LinkNeighborLoader(train_data, batch_size=2048, neg_sampling_ratio=.5,
                            num_neighbors=[10, 10])

# Evaluation loaders (one datapoint corresponds to a graph)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GraphSAGE(
    in_channels=train_dataset.num_features,
    hidden_channels=64,
    num_layers=3,
    out_channels=64,
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)


def train():
    model.train()

    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        link_pred = (out[data.edge_label_index[0]] *
                     out[data.edge_label_index[1]]).sum(dim=-1)
        loss = F.binary_cross_entropy_with_logits(link_pred, data.edge_label)
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * out.size(0)
    return total_loss / len(loader.dataset)


def get_representations_and_labels(graph_loader):
    xs, ys = [], []
    for data in graph_loader:
        ys.append(data.y)
        x, edge_index = data.x.to(device), data.edge_index.to(device)
        out = model(x, edge_index)
        xs.append(out)
    return torch.cat(xs, dim=0).cpu(), torch.cat(ys, dim=0).cpu()


@torch.no_grad()
def test(train_loader, val_loader, test_loader):
    model.eval()

    # Create classifier
    clf = MultiOutputClassifier(SGDClassifier(loss="log", penalty="l2"))

    # Train classifier on train data
    train_embeddings, train_labels = get_representations_and_labels(
        train_loader)
    clf.fit(train_embeddings, train_labels)
    train_predictions = clf.predict(train_embeddings)
    train_f1 = f1_score(train_labels, train_predictions, average='micro')

    # Evaluate on validation set
    val_embeddings, val_labels = get_representations_and_labels(val_loader)
    val_predictions = clf.predict(val_embeddings)
    val_f1 = f1_score(val_labels, val_predictions, average='micro')

    # Evaluate on testing set
    test_embeddings, test_labels = get_representations_and_labels(test_loader)
    test_predictions = clf.predict(test_embeddings)
    test_f1 = f1_score(test_labels, test_predictions, average='micro')

    return train_f1, val_f1, test_f1


for epoch in range(1, 51):
    loss = train()
    if epoch % 5 == 0:
        train_f1, val_f1, test_f1 = test(train_loader, val_loader, test_loader)
        print('Epoch: {:03d}, Loss: {:.4f}, Train F1: {:.4f}, '
              'Val F1: {:.4f}, Test F1: {:.4f}'.format(epoch, loss, train_f1,
                                                       val_f1, test_f1))
    else:
        print('Epoch: {:03d}, Loss: {:.4f}'.format(epoch, loss))
