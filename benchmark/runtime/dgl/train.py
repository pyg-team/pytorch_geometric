import time

import torch
import torch.nn.functional as F
import dgl
from dgl import DGLGraph


def train_runtime(Net, data, epochs, device):
    g = DGLGraph(data.graph)
    g.set_n_initializer(dgl.init.zero_initializer)
    x = torch.tensor(data.features, dtype=torch.float, device=device)
    mask = torch.tensor(data.train_mask, dtype=torch.uint8, device=device)
    y = torch.tensor(data.labels, dtype=torch.long, device=device)[mask]
    g.add_edges(g.nodes(), g.nodes())
    norm = torch.pow(g.in_degrees().float(), -0.5)
    norm[torch.isinf(norm)] = 0
    g.ndata['norm'] = norm.unsqueeze(1).to(device)

    model = Net(g, x.size(1), data.num_labels).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(x)
        loss = F.nll_loss(out[mask], y)
        loss.backward()
        optimizer.step()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_end = time.perf_counter()

    return t_end - t_start
