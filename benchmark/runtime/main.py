from itertools import product
from pathlib import Path

import torch
from runtime.gat import GAT
from runtime.gcn import GCN
from runtime.rgcn import RGCN
from runtime.train import train_runtime

from torch_geometric.datasets import Entities, Planetoid

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root = Path.joinpath(Path(__file__).resolve().parent.parent, 'data')
Cora = Planetoid(root.joinpath('Cora'), 'Cora')
CiteSeer = Planetoid(root.joinpath('CiteSeer'), 'CiteSeer')
PubMed = Planetoid(root.joinpath('PubMed'), 'PubMed')
MUTAG = Entities(root.joinpath('EntitiesMUTAG'), 'MUTAG')

# One training run before we start tracking duration to warm up GPU.
model = GCN(Cora.num_features, Cora.num_classes)
train_runtime(model, Cora[0], epochs=200, device=device)

for d, Net in product([Cora, CiteSeer, PubMed], [GCN, GAT]):
    model = Net(d.num_features, d.num_classes)
    t = train_runtime(model, d[0], epochs=200, device=device)
    print(f'{d.__repr__()[:-2]} - {Net.__name__}: {t:.2f}s')

for d, Net in product([MUTAG], [RGCN]):
    model = Net(d[0].num_nodes, d.num_classes, d.num_relations)
    t = train_runtime(model, d[0], epochs=200, device=device)
    print(f'{d.__repr__()[:-2]} - {Net.__name__}: {t:.2f}s')
