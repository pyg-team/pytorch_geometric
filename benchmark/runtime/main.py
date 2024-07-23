import os.path as osp
from itertools import product

import torch
from runtime.gat import GAT
from runtime.gcn import GCN
from runtime.rgcn import RGCN
from runtime.train import train_runtime

from torch_geometric.datasets import Entities, Planetoid

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

root = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
Cora = Planetoid(osp.join(root, 'Cora'), 'Cora')
CiteSeer = Planetoid(osp.join(root, 'CiteSeer'), 'CiteSeer')
PubMed = Planetoid(osp.join(root, 'PubMed'), 'PubMed')
MUTAG = Entities(osp.join(root, 'EntitiesMUTAG'), 'MUTAG')

# One training run before we start tracking duration to warm up GPU.
model = GCN(Cora.num_features, Cora.num_classes)
train_runtime(model, Cora[0], epochs=200, device=device)

for d, Net in product([Cora, CiteSeer, PubMed], [GCN, GAT]):
    model = Net(d.num_features, d.num_classes)
    t = train_runtime(model, d[0], epochs=200, device=device)
    print(f'{str(d)[:-2]} - {Net.__name__}: {t:.2f}s')

for d, Net in product([MUTAG], [RGCN]):
    model = Net(d[0].num_nodes, d.num_classes, d.num_relations)
    t = train_runtime(model, d[0], epochs=200, device=device)
    print(f'{str(d)[:-2]} - {Net.__name__}: {t:.2f}s')
