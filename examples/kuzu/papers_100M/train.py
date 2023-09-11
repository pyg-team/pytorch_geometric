import multiprocessing as mp
import os.path as osp

import kuzu
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import MLP, BatchNorm, SAGEConv

NUM_EPOCHS = 1
LOADER_BATCH_SIZE = 1024

print('Batch size:', LOADER_BATCH_SIZE)
print('Number of epochs:', NUM_EPOCHS)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Load the train set:
train_path = osp.join('.', 'papers100M-bin', 'split', 'time', 'train.csv.gz')
train_df = pd.read_csv(
    osp.abspath(train_path),
    compression='gzip',
    header=None,
)
input_nodes = torch.tensor(train_df[0].values, dtype=torch.long)

########################################################################
# The below code sets up the remote backend of Kùzu for PyG.
# Please refer to: https://kuzudb.com/docs/client-apis/python-api/overview.html
# for how to use the Python API of Kùzu.
########################################################################

# The buffer pool size of Kùzu is set to 40GB. You can change it to a smaller
# value if you have less memory.
KUZU_BM_SIZE = 40 * 1024**3

# Create Kùzu database:
db = kuzu.Database(osp.abspath(osp.join('.', 'papers100M')), KUZU_BM_SIZE)

# Get remote backend for PyG:
feature_store, graph_store = db.get_torch_geometric_remote_backend(
    mp.cpu_count())

# Plug the graph store and feature store into the `NeighborLoader`.
# Note that `filter_per_worker` is set to `False`. This is because the Kùzu
# database is already using multi-threading to scan the features in parallel
# and the database object is not fork-safe.
loader = NeighborLoader(
    data=(feature_store, graph_store),
    num_neighbors={('paper', 'cites', 'paper'): [12, 12, 12]},
    batch_size=LOADER_BATCH_SIZE,
    input_nodes=('paper', input_nodes),
    num_workers=4,
    filter_per_worker=False,
)


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=0.2):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns.append(BatchNorm(hidden_channels))
        for i in range(1, num_layers):
            self.layers.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(BatchNorm(hidden_channels))

        self.mlp = MLP(
            in_channels=in_channels + num_layers * hidden_channels,
            hidden_channels=2 * out_channels,
            out_channels=out_channels,
            num_layers=2,
            norm='batch_norm',
            act='leaky_relu',
        )

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        xs = [x]
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = x.relu()
            x = F.dropout(x, p=self.dropout, training=self.training)
            xs.append(x)
        return self.mlp(torch.cat(xs, dim=-1))


model = GraphSAGE(in_channels=128, hidden_channels=1024, out_channels=172,
                  num_layers=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

for epoch in range(1, NUM_EPOCHS + 1):
    total_loss = total_examples = 0
    for batch in tqdm(loader):
        batch = batch.to(device)
        batch_size = batch['paper'].batch_size

        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)[:batch_size]
        y = batch.y[:batch_size].long().view(-1)
        loss = F.cross_entropy_loss(out, y)

        loss.backward()
        optimizer.step()

        total_loss += float(loss) * y.numel()
        total_examples += y.numel()

    print(f'Epoch: {epoch:02d}, Loss: {total_loss / total_examples:.4f}')
