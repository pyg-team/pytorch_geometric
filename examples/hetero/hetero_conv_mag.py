import torch
import torch.nn.functional as F
from tqdm import tqdm

import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import HeteroConv, Linear, SAGEConv
from torch_geometric.utils import trim_to_layer

use_sparse_tensor = False
trim = True

transform = T.ToSparseTensor(
    remove_edge_index=False) if use_sparse_tensor else None
if transform is None:
    transform = T.ToUndirected(merge=True)
else:
    transform = T.Compose([T.ToUndirected(merge=True), transform])

dataset = OGB_MAG(root='../../data', preprocess='metapath2vec',
                  transform=transform)

data = dataset[0]
device = 'cpu'


class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers, trim):
        super().__init__()
        self.trim = trim
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    ('paper', 'cites', 'paper'):
                    SAGEConv((-1, -1), hidden_channels),
                    ('author', 'writes', 'paper'):
                    SAGEConv((-1, -1), hidden_channels),
                    ('paper', 'rev_writes', 'author'):
                    SAGEConv((-1, -1), hidden_channels),
                    ('author', 'affiliated_with', 'institution'):
                    SAGEConv((-1, -1), hidden_channels),
                    ('institution', 'rev_affiliated_with', 'author'):
                    SAGEConv((-1, -1), hidden_channels),
                    ('paper', 'has_topic', 'field_of_study'):
                    SAGEConv((-1, -1), hidden_channels),
                    ('field_of_study', 'rev_has_topic', 'paper'):
                    SAGEConv((-1, -1), hidden_channels),
                }, aggr='sum')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict, num_sampled_edges=None,
                num_sampled_nodes=None):

        for i, conv in enumerate(self.convs):

            if not use_sparse_tensor and self.trim:
                x_dict, edge_index_dict = trim_to_layer(
                    layer=i, node_attrs=x_dict, edge_index=edge_index_dict,
                    num_nodes_per_layer=num_sampled_nodes,
                    num_edges_per_layer=num_sampled_edges)

            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        return self.lin(x_dict['paper'])


model = HeteroGNN(hidden_channels=64, out_channels=dataset.num_classes,
                  num_layers=3, trim=trim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

kwargs = {'batch_size': 1024, 'num_workers': 0}
train_loader = NeighborLoader(data, num_neighbors=[10] * 3, shuffle=True,
                              input_nodes=('paper', data['paper'].train_mask),
                              **kwargs)

val_loader = NeighborLoader(data, num_neighbors=[10] * 3, shuffle=False,
                            input_nodes=('paper', data['paper'].val_mask),
                            **kwargs)


def train():
    model.train()

    total_examples = total_loss = 0
    for batch in tqdm(train_loader):
        edge_index_dict = (batch.adj_t_dict
                           if use_sparse_tensor else batch.edge_index_dict)
        optimizer.zero_grad()
        batch_size = batch['paper'].batch_size
        if trim:
            out = model(batch.x_dict, edge_index_dict,
                        num_sampled_edges=batch.num_sampled_edges_dict,
                        num_sampled_nodes=batch.num_sampled_nodes_dict)
        else:
            out = model(batch.x_dict, edge_index_dict)

        loss = F.cross_entropy(out[:batch_size], batch['paper'].y[:batch_size])
        loss.backward()
        optimizer.step()

        total_examples += batch_size
        total_loss += float(loss) * batch_size

    return total_loss / total_examples


@torch.no_grad()
def test(loader):
    model.eval()

    total_examples = total_correct = 0
    for batch in tqdm(loader):
        batch = batch.to(device, 'edge_index')
        batch_size = batch['paper'].batch_size
        edge_index_dict = (batch.adj_t_dict
                           if use_sparse_tensor else batch.edge_index_dict)
        if trim:
            out = model(
                batch.x_dict, edge_index_dict,
                num_sampled_edges=batch.num_sampled_edges_dict,
                num_sampled_nodes=batch.num_sampled_nodes_dict)[:batch_size]
        else:
            out = model(batch.x_dict, edge_index_dict)[:batch_size]

        pred = out.argmax(dim=-1)

        total_examples += batch_size
        total_correct += int((pred == batch['paper'].y[:batch_size]).sum())

    return total_correct / total_examples


for epoch in range(1, 6):
    print("Train")
    loss = train()
    print("Test")
    val_acc = test(val_loader)
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val: {val_acc:.4f}')
