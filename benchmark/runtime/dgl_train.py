import math
import torch
from torch.nn import Parameter, Linear
import torch.nn.functional as F
import dgl.function as fn
import argparse
import time
from dgl import DGLGraph
from dgl.data import load_data
from dgl.nn.pytorch import EdgeSoftmax

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--model', type=str, default='gcn')
args = parser.parse_args()

data = load_data(args)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
features = torch.tensor(data.features, dtype=torch.float, device=device)
labels = torch.tensor(data.labels, dtype=torch.long, device=device)
train_mask = torch.tensor(data.train_mask, dtype=torch.uint8, device=device)
in_feats = features.shape[1]
n_classes = data.num_labels

g = DGLGraph(data.graph)
g.add_edges(g.nodes(), g.nodes())
degs = g.in_degrees().float()
norm = torch.pow(degs, -0.5)
norm[torch.isinf(norm)] = 0
g.ndata['norm'] = norm.unsqueeze(1).to(device)

if args.model == 'gcn':
    model = GCN(g, in_feats, n_classes).to(device)
elif args.model == 'gat':
    model = GAT(g, in_feats, n_classes).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
y = labels[train_mask]

if torch.cuda.is_available():
    torch.cuda.synchronize()
t_start = time.perf_counter()
for epoch in range(200):
    model.train()
    logits = model(features)
    loss = F.nll_loss(logits[train_mask], y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
if torch.cuda.is_available():
    torch.cuda.synchronize()
t_end = time.perf_counter()
print(t_end - t_start)
