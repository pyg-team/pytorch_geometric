from typing import List, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import HeteroConv, Linear, SAGEConv
from torch_geometric.typing import Adj, OptPairTensor, Size
from torch_geometric.utils import trim_adj, trim_x

use_sparse_tensor = True
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


class HierarchicalSAGEConv(SAGEConv):
    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None,
                num_sampled_nodes_per_hop: Union[List[int],
                                                 Tuple[List[int]]] = None,
                num_sampled_edges_per_hop: Union[List[int],
                                                 Tuple[List[int]]] = None,
                layer: int = None, edge_index_key: str = None,
                edge_indexes: dict = None) -> Tensor:
        if num_sampled_nodes_per_hop and num_sampled_edges_per_hop:
            edge_indexes[edge_index_key] = trim_adj(edge_index, layer,
                                                    num_sampled_nodes_per_hop,
                                                    num_sampled_edges_per_hop)
            layer += 1  # linear layer trim opt
            edge_index = edge_indexes[edge_index_key]

            if edge_index.numel() == 0:
                x = x[1] if isinstance(x, Tuple) else x
                return trim_x(x, layer, num_sampled_nodes_per_hop)

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        if self.project and hasattr(self, 'lin'):
            x = (self.lin(x[0]).relu(), x[1])

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)

        x_r = x[1]

        # linear layer trim opt
        if num_sampled_nodes_per_hop and num_sampled_edges_per_hop:
            out = trim_x(out, layer, num_sampled_nodes_per_hop)
            x_r = trim_x(
                x_r, layer,
                num_sampled_nodes_per_hop) if self.root_weight else None

        out = self.lin_l(out)

        if self.root_weight and x_r is not None:
            out = out + self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out


class HierarchicalHeteroGraphSage(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers, trim):
        super().__init__()
        self.trim = trim
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    ('paper', 'cites', 'paper'):
                    HierarchicalSAGEConv((-1, -1), hidden_channels),
                    ('author', 'writes', 'paper'):
                    HierarchicalSAGEConv((-1, -1), hidden_channels),
                    ('paper', 'rev_writes', 'author'):
                    HierarchicalSAGEConv((-1, -1), hidden_channels),
                    ('author', 'affiliated_with', 'institution'):
                    HierarchicalSAGEConv((-1, -1), hidden_channels),
                    ('institution', 'rev_affiliated_with', 'author'):
                    HierarchicalSAGEConv((-1, -1), hidden_channels),
                    ('paper', 'has_topic', 'field_of_study'):
                    HierarchicalSAGEConv((-1, -1), hidden_channels),
                    ('field_of_study', 'rev_has_topic', 'paper'):
                    HierarchicalSAGEConv((-1, -1), hidden_channels),
                }, aggr='sum')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict, num_sampled_edges_dict=None,
                num_sampled_nodes_dict=None):

        for i, conv in enumerate(self.convs):
            kwargs_dict = {}

            if self.trim:
                kwargs_dict.update(
                    {"num_sampled_edges_per_hop_dict": num_sampled_edges_dict})
                kwargs_dict.update(
                    {"num_sampled_nodes_per_hop_dict": num_sampled_nodes_dict})
                layer = {k: i for k in edge_index_dict.keys()}
                kwargs_dict.update({"layer_dict": layer})
                edge_index_key = {k: k for k in edge_index_dict.keys()}
                kwargs_dict.update({"edge_index_key_dict": edge_index_key})
                edge_indexes = {
                    k: edge_index_dict
                    for k in edge_index_dict.keys()
                }
                kwargs_dict.update({"edge_indexes_dict": edge_indexes})

            x_dict = conv(x_dict, edge_index_dict, **kwargs_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        return self.lin(x_dict['paper'])


model = HierarchicalHeteroGraphSage(hidden_channels=64,
                                    out_channels=dataset.num_classes,
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
                        num_sampled_nodes_dict=batch.num_sampled_nodes_dict,
                        num_sampled_edges_dict=batch.num_sampled_edges_dict)
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
            out = model(batch.x_dict, edge_index_dict,
                        num_sampled_nodes_dict=batch.num_sampled_nodes_dict,
                        num_sampled_edges_dict=batch.num_sampled_edges_dict
                        )[:batch_size]
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
