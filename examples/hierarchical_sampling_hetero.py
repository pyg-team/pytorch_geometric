import argparse
from time import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import HeteroConv, Linear, SAGEConv
from torch_geometric.utils._trim_to_layer import TrimToLayer

_trim_to_layer = TrimToLayer()

parser = argparse.ArgumentParser()
parser.add_argument('--use-sparse-tensor', action='store_true')
parser.add_argument('--hgam', action='store_true')
parser.add_argument('--batch-size', type=int, default=512)  #512
parser.add_argument('--hidden-dim', type=int, default=256)
parser.add_argument('--num-layers', type=int, default=3)
parser.add_argument('--num-neighbors', type=int, default=10)  
parser.add_argument('--num-epochs', type=int, default=5)
args = parser.parse_args()
args.use_sparse_tensor = True
args.hgam = True
print(args)

transforms = [T.ToUndirected(merge=True)]
if args.use_sparse_tensor:
    transforms.append(T.ToSparseTensor())
dataset = OGB_MAG(root='../data/', preprocess='metapath2vec',
                  transform=T.Compose(transforms))
data = dataset[0]
print("\nDataset in use is OGB_MAG:\n")
print(data)
node_types, edge_types = data.metadata()
print(node_types)
print(edge_types)


class HierarchicalHeteroGraphSage(torch.nn.Module):
    def __init__(self, edge_types, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    edge_type: SAGEConv((-1, -1), hidden_channels)
                    for edge_type in edge_types
                }, aggr='sum')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict, num_sampled_edges_dict,
                num_sampled_nodes_dict):

        for i, conv in enumerate(self.convs):
            if args.hgam:
                x_dict, xrr_dict, edge_index_dict, _ = _trim_to_layer(
                    layer=i,
                    num_sampled_nodes_per_hop=num_sampled_nodes_dict,
                    num_sampled_edges_per_hop=num_sampled_edges_dict,
                    x=x_dict,
                    edge_index=edge_index_dict,
                )

            if args.hgam:
                x_dict = conv(x_dict, edge_index_dict, xra_dict=xrr_dict)
            else:
                x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        return self.lin(x_dict['paper'])


model = HierarchicalHeteroGraphSage(
    edge_types=data.edge_types,
    hidden_channels=args.hidden_dim,
    out_channels=dataset.num_classes,
    num_layers=args.num_layers,
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

kwargs = {'batch_size': args.batch_size, 'num_workers': 0}
train_loader = NeighborLoader(
    data,
    num_neighbors={key: [1] * args.num_layers
                   for key in data.edge_types},
    shuffle=True,
    input_nodes=('paper', data['paper'].train_mask),
    **kwargs,
)
print("train_loader is: ", train_loader)


def train():
    model.train()

    total_examples = total_loss = 0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()

        out = model(
            batch.x_dict,
            batch.adj_t_dict
            if args.use_sparse_tensor else batch.edge_index_dict,
            num_sampled_nodes_dict=batch.num_sampled_nodes_dict
            if args.hgam else None,
            num_sampled_edges_dict=batch.num_sampled_edges_dict
            if args.hgam else None,
        )

        batch_size = batch['paper'].batch_size
        loss = F.cross_entropy(out[:batch_size], batch['paper'].y[:batch_size])

        loss.backward()
        optimizer.step()

        total_examples += batch_size
        total_loss += float(loss) * batch_size

    avg_loss = total_loss / total_examples
    print("hierarchical_sampling: avg_loss is: ", avg_loss)
    return avg_loss


epoch_time = []
for epoch in range(args.num_epochs):
    epoch_beg = time()
    loss = train()
    epoch_time.append(time() - epoch_beg)
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')

print(epoch_time)
print(f'Mean time: {np.mean(epoch_time)}, std: {np.std(epoch_time)}')
