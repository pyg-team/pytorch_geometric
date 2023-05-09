import multiprocessing as mp
import time
from os import path

import kuzu
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import MLP
from torch_geometric.nn.conv import SAGEConv

NUM_EPOCHS = 1
LOADER_BATCH_SIZE = 1024

print("Batch size:", LOADER_BATCH_SIZE)
print("Number of epochs:", NUM_EPOCHS)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Load the train set
train_df = pd.read_csv(
    path.abspath(
        path.join('.', 'papers100M-bin', 'split', 'time', 'train.csv.gz')),
    compression='gzip', header=None)
input_nodes = torch.tensor(train_df[0].values, dtype=torch.long)

########################################################################
# The below code sets up the remote backend of Kùzu for PyG.
# Please refer to: https://kuzudb.com/docs/client-apis/python-api/overview.html
# for how to use the Python API of Kùzu.
########################################################################
# The buffer pool size of Kùzu is set to 40GB. You can change it to a smaller
# value if you have less memory.
KUZU_BM_SIZE = 40 * 1024**3
# Create Kùzu database
db = kuzu.Database(path.abspath(path.join(".", "papers100M")), KUZU_BM_SIZE)
# Get remote backend for torch geometric
feature_store, graph_store = db.get_torch_geometric_remote_backend(
    mp.cpu_count())
# Plug the graph store and feature store into the NeighborLoader.
# Note that `filter_per_worker` is set to False. This is because Kùzu database
# is already using multi-threading to scan the features in parallel and the
# database object is not fork-safe.
loader_kuzu = NeighborLoader(
    data=(feature_store, graph_store),
    num_neighbors={('paper', 'cites', 'paper'): [12, 12, 12]},
    batch_size=LOADER_BATCH_SIZE,
    input_nodes=('paper', input_nodes),
    num_workers=4,
    filter_per_worker=False,
)


class SAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation,
                 dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.layers.append(
            SAGEConv(in_feats, n_hidden, aggr='mean', bias=False))
        self.bns.append(torch.nn.BatchNorm1d(n_hidden))
        for i in range(1, n_layers - 1):
            self.layers.append(
                SAGEConv(n_hidden, n_hidden, aggr='mean', bias=False))
            self.bns.append(torch.nn.BatchNorm1d(n_hidden))
        self.layers.append(
            SAGEConv(n_hidden, n_hidden, aggr='mean', bias=False))
        self.bns.append(torch.nn.BatchNorm1d(n_hidden))
        self.mlp = MLP(
            in_channels=in_feats + n_hidden * n_layers,
            hidden_channels=2 * n_classes,
            out_channels=n_classes,
            num_layers=2,
            norm="batch_norm",
            act='leaky_relu',
        )
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.profile = locals()

    def forward(self, edge_list, x):
        collect = [x]
        h = x
        h = self.dropout(h)
        for layer_index, layer in enumerate(self.layers):
            h = layer(h, edge_list)
            h = self.bns[layer_index](h)
            h = self.activation(h)
            h = self.dropout(h)
            collect.append(h)
        collect = torch.cat(collect, dim=1)
        return self.mlp(collect)


def count_parameters(model):
    return sum(
        [np.prod(p.size()) for p in model.parameters() if p.requires_grad])


model = SAGE(128, 1024, 172, 3, torch.nn.functional.relu, 0.2)
print("Model:", model)
print("Number of parameters: ", count_parameters(model))
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(NUM_EPOCHS):
    i = 0
    start_time = time.time()
    for batch in loader_kuzu:
        x = batch['paper']['x']
        y = batch['paper']['y']
        edge_index = batch['paper', 'cites', 'paper'].edge_index
        y = y[:batch['paper'].batch_size]
        y = y.long().flatten()
        x = x.to(device)
        y = y.to(device)
        edge_index = edge_index.to(device)
        train_start_time = time.time()
        model.train()
        optimizer.zero_grad()
        out = model(edge_index, x)
        out = out[:batch['paper'].batch_size]
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        end_time = time.time()
        batch_time = end_time - start_time
        print("epoch:", epoch + 1, "\t", "batch:", i + 1, "\t", "time:",
              "%.2f" % batch_time, "\t"
              "loss:", "%.4f" % loss.item(), "\t")
        i += 1
        start_time = time.time()
