from itertools import product

import torch
from dgl.data import citation_graph

from runtime.dgl.gcn import GCN, GCNSPMV
from runtime.dgl.gat import GAT, GATImproved
from runtime.dgl.train import train_runtime
from runtime.dgl.hidden import HiddenPrint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with HiddenPrint():
    Cora = citation_graph.load_cora()
    CiteSeer = citation_graph.load_citeseer()
    PubMed = citation_graph.load_pubmed()

# One training run before we start tracking duration to warm up GPU.
train_runtime(GCNSPMV, Cora, epochs=200, device=device)

for d, Net in product([Cora, CiteSeer, PubMed],
                      [GCN, GCNSPMV, GAT, GATImproved]):
    t = train_runtime(Net, d, epochs=200, device=device)
    print('{} - {}: {:.2f}s'.format(d.name, Net.__name__, t))
