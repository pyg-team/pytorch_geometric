import argparse
import copy
import os.path as osp

import torch
from sklearn.linear_model import LogisticRegression

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import RECT_L

# RECT focuses on the zero-shot, i.e. completely-imbalanced label setting:
# For this, we first remove "unseen" classes from the training set and train a
# RECT (or more specifically its supervised part RECT-L) model in the zero-shot
# label scenario. Lastly, we train a simple classifier to evaluate the final
# performance of the embeddings based on the original labels.

# Datasets              Citeseer             Cora          Pubmed
# Unseen Classes  [1, 2, 5]  [3, 4]  [1, 2, 3]  [3, 4, 6]  [2]
# RECT-L          66.30      68.20   74.60      71.20      75.30
# GCN             51.80      55.70   55.80      57.10      59.80
# NodeFeats       61.40      61.40   57.50      57.50      73.10

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora',
                    choices=['Cora', 'CiteSeer', 'PubMed'])
parser.add_argument('--unseen-classes', type=int, nargs='*', default=[1, 2, 3])
args = parser.parse_args()

path = osp.join(osp.dirname(osp.realpath(__file__)), '../data/Planetoid')
train_mask_original = Planetoid(path, args.dataset)[0].train_mask.clone()
transform = T.Compose([
    T.NormalizeFeatures(),
    T.SVDFeatureReduction(200),
    T.GDC(),
])
dataset = Planetoid(path, args.dataset, transform=transform)
data = dataset[0]
zs_data = T.RemoveTrainingClasses(args.unseen_classes)(copy.copy(data))

model = RECT_L(200, 200, normalize=False, dropout=0.0)
zs_data.y = model.get_semantic_labels(zs_data.x, zs_data.y, zs_data.train_mask)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, zs_data = model.to(device), zs_data.to(device)

criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

model.train()
for epoch in range(1, 201):
    optimizer.zero_grad()
    out = model(zs_data.x, zs_data.edge_index, zs_data.edge_attr)
    loss = criterion(out[zs_data.train_mask], zs_data.y)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch:03d}, Loss {loss:.4f}')

model.eval()
with torch.no_grad():
    h = model.embed(zs_data.x, zs_data.edge_index, zs_data.edge_attr).cpu()

reg = LogisticRegression()
reg.fit(h[data.train_mask].numpy(), data.y[data.train_mask].numpy())
test_acc = reg.score(h[data.test_mask].numpy(), data.y[data.test_mask].numpy())
print(f'Test Acc: {test_acc:.4f}')
