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
train_loader = NeighborLoader(data, input_nodes=data.train_mask,
                              num_neighbors=[25, 10], shuffle=True, **kwargs)


def train(model, loader, optimizer, device, progress_bar=True, desc="", trim=False):
    if progress_bar:
        loader = tqdm(loader, desc=desc)
    model.train()
    for batch in loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        if hasattr(batch, 'adj_t'):
            edge_index = batch.adj_t
        else:
            edge_index = batch.edge_index
        if not trim:
            out = model(batch.x, edge_index)
        else:
            out = model( 
                batch.x,
                edge_index,
                num_sampled_nodes_per_hop=batch.num_sampled_nodes,
                num_sampled_edges_per_hop=batch.num_sampled_edges,
            )
        batch_size = batch.batch_size
        out = out[:batch_size]
        target = batch.y[:batch_size]
        loss = F.cross_entropy(out, target)
        loss.backward()
        optimizer.step()


model = GraphSAGE(data.num_node_features, hidden_channels=64,
                  num_layers=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loader = train_loader

print("One Epoch training without Hierarchical Graph Sampling")
train(model, loader, optimizer, device, progress_bar=True, desc="", trim=False)

print("One Epoch training WITH Hierarchical Graph Sampling")
train(model, loader, optimizer, device, progress_bar=True, desc="", trim=True)
