import os.path as osp

import torch
import torch.nn.functional as F
from tqdm import tqdm

import torch_geometric.transforms as T
from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn.models.basic_gnn import GraphSAGE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Reddit')
dataset = Reddit(path, transform=T.ToSparseTensor())

# Already send node features/labels to GPU for faster access during sampling:
data = dataset[0].to(device, 'x', 'y')

use_sparse_tensor = True

kwargs = {'batch_size': 1, 'num_workers': 6, 'persistent_workers': True} #1024
loader = NeighborLoader(data, input_nodes=data.train_mask,
                        num_neighbors=[10, 5, 5], shuffle=True, **kwargs)

model = GraphSAGE(
    dataset.num_features,
    hidden_channels=64,
    out_channels=dataset.num_classes,
    num_layers=3,
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train(trim=False):
    total_examples = total_loss = 0
    for batch in tqdm(loader):
        optimizer.zero_grad()
        batch = batch.to(device)

        edge_index = batch.adj_t if use_sparse_tensor else batch.edge_index

        if not trim:
            out = model(batch.x, edge_index)
            out = out[:batch.
                      batch_size]  # without trim_rect_adj_mat we compute more representations than necessary
        else:
            out = model(
                batch.x,
                edge_index,
                num_sampled_nodes_per_hop=batch.num_sampled_nodes,
                num_sampled_edges_per_hop=batch.num_sampled_edges,
            )
        y = batch.y[:batch.batch_size]

        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()

        total_examples += batch.batch_size
        total_loss += float(loss) * batch.batch_size

    avg_loss = total_loss / total_examples
    print("avg_loss is: ", avg_loss)


# print('\nOne epoch training without Hierarchical Graph Sampling:')
# train(trim=False)

print('\nOne epoch training with Hierarchical Graph Sampling:')
train(trim=True)
