import argparse
import os.path as osp

import torch
import torch.nn.functional as F
from tqdm import tqdm

import torch_geometric.transforms as T
from torch_geometric import EdgeIndex
from torch_geometric.datasets import MovieLens
from torch_geometric.loader import LinkNeighborLoader, NeighborLoader
from torch_geometric.metrics import (
    LinkPredMAP,
    LinkPredPrecision,
    LinkPredRecall,
)
from torch_geometric.nn import MIPSKNNIndex, SAGEConv, to_hetero

parser = argparse.ArgumentParser()
parser.add_argument('--k', type=int, default=20, help='Number of predictions')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = osp.join(osp.dirname(osp.realpath(__file__)), '../../data/MovieLens')
data = MovieLens(path, model_name='all-MiniLM-L6-v2')[0]

# Add user node features for message passing:
data['user'].x = torch.eye(data['user'].num_nodes)
del data['user'].num_nodes

# Only use edges with high ratings (>= 4):
mask = data['user', 'rates', 'movie'].edge_label >= 4
data['user', 'movie'].edge_index = data['user', 'movie'].edge_index[:, mask]
data['user', 'movie'].time = data['user', 'movie'].time[mask]
del data['user', 'movie'].edge_label  # Drop rating information from graph.

# Add a reverse ('movie', 'rev_rates', 'user') relation for message passing:
data = T.ToUndirected()(data)

# Perform a temporal link-level split into training and test edges:
edge_label_index = data['user', 'movie'].edge_index
time = data['user', 'movie'].time

perm = time.argsort()
train_index = perm[:int(0.8 * perm.numel())]
test_index = perm[int(0.8 * perm.numel()):]

kwargs = dict(  # Shared data loader arguments:
    data=data,
    num_neighbors=[5, 5, 5],
    batch_size=256,
    time_attr='time',
    num_workers=4,
    persistent_workers=True,
    temporal_strategy='last',
)

train_loader = LinkNeighborLoader(
    edge_label_index=(('user', 'movie'), edge_label_index[:, train_index]),
    edge_label_time=time[train_index] - 1,  # No leakage.
    neg_sampling=dict(mode='binary', amount=2),
    shuffle=True,
    **kwargs,
)

# During testing, we sample node-level subgraphs from both endpoints to
# retrieve their embeddings.
# This allows us to do efficient k-NN search on top of embeddings:
src_loader = NeighborLoader(
    input_nodes='user',
    input_time=(time[test_index].min() - 1).repeat(data['user'].num_nodes),
    **kwargs,
)
dst_loader = NeighborLoader(
    input_nodes='movie',
    input_time=(time[test_index].min() - 1).repeat(data['movie'].num_nodes),
    **kwargs,
)

# Save test edges and the edges we want to exclude when evaluating:
sparse_size = (data['user'].num_nodes, data['movie'].num_nodes)
test_edge_label_index = EdgeIndex(
    edge_label_index[:, test_index].to(device),
    sparse_size=sparse_size,
).sort_by('row')[0]
test_exclude_links = EdgeIndex(
    edge_label_index[:, train_index].to(device),
    sparse_size=sparse_size,
).sort_by('row')[0]


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels)
        self.conv3 = SAGEConv((-1, -1), hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index)
        return x


class InnerProductDecoder(torch.nn.Module):
    def forward(self, x_dict, edge_label_index):
        x_src = x_dict['user'][edge_label_index[0]]
        x_dst = x_dict['movie'][edge_label_index[1]]
        return (x_src * x_dst).sum(dim=-1)


class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.encoder = GNN(hidden_channels)
        self.encoder = to_hetero(self.encoder, data.metadata(), aggr='sum')
        self.decoder = InnerProductDecoder()

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        x_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(x_dict, edge_label_index)


model = Model(hidden_channels=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train():
    model.train()

    total_loss = total_examples = 0
    for batch in tqdm(train_loader):
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(
            batch.x_dict,
            batch.edge_index_dict,
            batch['user', 'movie'].edge_label_index,
        )
        y = batch['user', 'movie'].edge_label

        loss = F.binary_cross_entropy_with_logits(out, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * y.numel()
        total_examples += y.numel()

    return total_loss / total_examples


@torch.no_grad()
def test(edge_label_index, exclude_links):
    model.eval()

    dst_embs = []
    for batch in dst_loader:  # Collect destination node/movie embeddings:
        batch = batch.to(device)
        emb = model.encoder(batch.x_dict, batch.edge_index_dict)['movie']
        emb = emb[:batch['movie'].batch_size]
        dst_embs.append(emb)
    dst_emb = torch.cat(dst_embs, dim=0)
    del dst_embs

    # Instantiate k-NN index based on maximum inner product search (MIPS):
    mips = MIPSKNNIndex(dst_emb)

    # Initialize metrics:
    map_metric = LinkPredMAP(k=args.k).to(device)
    precision_metric = LinkPredPrecision(k=args.k).to(device)
    recall_metric = LinkPredRecall(k=args.k).to(device)

    num_processed = 0
    for batch in src_loader:  # Collect source node/user embeddings:
        batch = batch.to(device)

        # Compute user embeddings:
        emb = model.encoder(batch.x_dict, batch.edge_index_dict)['user']
        emb = emb[:batch['user'].batch_size]

        # Filter labels/exclusion by current batch:
        _edge_label_index = edge_label_index.sparse_narrow(
            dim=0,
            start=num_processed,
            length=emb.size(0),
        )
        _exclude_links = exclude_links.sparse_narrow(
            dim=0,
            start=num_processed,
            length=emb.size(0),
        )
        num_processed += emb.size(0)

        # Perform MIPS search:
        _, pred_index_mat = mips.search(emb, args.k, _exclude_links)

        # Update retrieval metrics:
        map_metric.update(pred_index_mat, _edge_label_index)
        precision_metric.update(pred_index_mat, _edge_label_index)
        recall_metric.update(pred_index_mat, _edge_label_index)

    return (
        float(map_metric.compute()),
        float(precision_metric.compute()),
        float(recall_metric.compute()),
    )


for epoch in range(1, 16):
    train_loss = train()
    print(f'Epoch: {epoch:02d}, Loss: {train_loss:.4f}')
    val_map, val_precision, val_recall = test(
        test_edge_label_index,
        test_exclude_links,
    )
    print(f'Test MAP@{args.k}: {val_map:.4f}, '
          f'Test Precision@{args.k}: {val_precision:.4f}, '
          f'Test Recall@{args.k}: {val_recall:.4f}')
