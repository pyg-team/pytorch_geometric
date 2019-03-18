from itertools import product

import torch

from runtime.dgl.gcn import GCN
from runtime.dgl.gat import GAT
from runtime.dgl.train import train_runtime

from dgl.data import citation_graph
from dgl import DGLGraph

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Cora = citation_graph.load_cora()
CiteSeer = citation_graph.load_citeseer()
PubMed = citation_graph.load_pubmed()

for d, Net in product([Cora, CiteSeer, PubMed], [GCN, GAT]):
    g = DGLGraph(d.graph)
    x = torch.tensor(d.features, dtype=torch.float, device=device)
    y = torch.tensor(d.labels, dtype=torch.long, device=device)
    mask = torch.tensor(d.train_mask, dtype=torch.uint8, device=device)
    g.add_edges(g.nodes(), g.nodes())
    norm = torch.pow(g.in_degrees().float(), -0.5)
    norm[torch.isinf(norm)] = 0
    g.ndata['norm'] = norm.unsqueeze(1).to(device)
    model = Net(g, x.size(1), d.num_labels)
    t = train_runtime(model, x, y, mask, epochs=200, device=device)
    print('{} - {}: {:.2f}s'.format(d.name, Net.__name__, t))
