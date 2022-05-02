import os.path as osp

import torch
import torch.nn.functional as F
import tqdm
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.multioutput import MultiOutputClassifier

from torch_geometric.data import Batch
from torch_geometric.datasets import PPI
from torch_geometric.loader import DataLoader, LinkNeighborLoader
from torch_geometric.nn import GraphSAGE

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'PPI')
train_dataset = PPI(path, split='train')
val_dataset = PPI(path, split='val')
test_dataset = PPI(path, split='test')

# Group all training graphs into a single graph to perform sampling:
train_data = Batch.from_data_list(train_dataset)
loader = LinkNeighborLoader(train_data, batch_size=2048, shuffle=True,
                            neg_sampling_ratio=0.5, num_neighbors=[10, 10],
                            num_workers=6, persistent_workers=True)

# Evaluation loaders (one datapoint corresponds to a graph)
train_loader = DataLoader(train_dataset, batch_size=2)
val_loader = DataLoader(val_dataset, batch_size=2)
test_loader = DataLoader(test_dataset, batch_size=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GraphSAGE(
    in_channels=train_dataset.num_features,
    hidden_channels=64,
    num_layers=2,
    out_channels=64,
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)


def train():
    model.train()

    total_loss = total_examples = 0
    for data in tqdm.tqdm(loader):
        data = data.to(device)
        optimizer.zero_grad()
        h = model(data.x, data.edge_index)

        h_src = h[data.edge_label_index[0]]
        h_dst = h[data.edge_label_index[1]]
        link_pred = (h_src * h_dst).sum(dim=-1)  # Inner product.

        loss = F.binary_cross_entropy_with_logits(link_pred, data.edge_label)
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * link_pred.numel()
        total_examples += link_pred.numel()

    return total_loss / total_examples


@torch.no_grad()
def encode(loader):
    model.eval()

    xs, ys = [], []
    for data in loader:
        data = data.to(device)
        xs.append(model(data.x, data.edge_index).cpu())
        ys.append(data.y.cpu())
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


@torch.no_grad()
def test():
    # Train classifier on training set:
    x, y = encode(train_loader)

    clf = MultiOutputClassifier(SGDClassifier(loss='log', penalty='l2'))
    clf.fit(x, y)

    train_f1 = f1_score(y, clf.predict(x), average='micro')

    # Evaluate on validation set:
    x, y = encode(val_loader)
    val_f1 = f1_score(y, clf.predict(x), average='micro')

    # Evaluate on test set:
    x, y = encode(test_loader)
    test_f1 = f1_score(y, clf.predict(x), average='micro')

    return train_f1, val_f1, test_f1


for epoch in range(1, 6):
    loss = train()
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
    train_f1, val_f1, test_f1 = test()
    print(f'Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}, '
          f'Test F1: {test_f1:.4f}')
