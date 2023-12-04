import os.path as osp

import torch
import torch.nn.functional as F
from tqdm import tqdm

from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn.models.basic_gnn import GraphSAGE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Reddit')
dataset = Reddit(path)

# Already send node features/labels to GPU for faster access during sampling:
data = dataset[0].to(device, 'x', 'y')

kwargs = {'batch_size': 1024, 'num_workers': 6, 'persistent_workers': True}
loader = NeighborLoader(data, input_nodes=data.train_mask,
                        num_neighbors=[20, 10, 5], shuffle=True, **kwargs)

model = GraphSAGE(
    dataset.num_features,
    hidden_channels=64,
    out_channels=dataset.num_classes,
    num_layers=3,
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train(trim=False):
    for batch in tqdm(loader):
        optimizer.zero_grad()
        batch = batch.to(device)

        if not trim:
            out = model(batch.x, batch.edge_index)
        else:
            out = model(
                batch.x,
                batch.edge_index,
                num_sampled_nodes_per_hop=batch.num_sampled_nodes,
                num_sampled_edges_per_hop=batch.num_sampled_edges,
            )

        out = out[:batch.batch_size]
        y = batch.y[:batch.batch_size]

        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()


print('One epoch training without Hierarchical Graph Sampling:')
train(trim=False)

print('One epoch training with Hierarchical Graph Sampling:')
train(trim=True)
