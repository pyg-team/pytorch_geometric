import copy

import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor
from tqdm import tqdm

import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import HeteroConv, Linear, SAGEConv
from torch_geometric.utils import trim_to_layer

use_sparse_tensor = False

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
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()
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
                num_sampled_nodes=None, trim=False):

        for i, conv in enumerate(self.convs):

            if not use_sparse_tensor and trim:
                x_dict, edge_index_dict = trim_to_layer(
                    layer=i, node_attrs=x_dict, edge_index=edge_index_dict,
                    num_edges_per_layer=num_sampled_edges,
                    num_nodes_per_layer=num_sampled_nodes)

            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        return self.lin(x_dict['paper'])


num_layers = 3

model = HeteroGNN(hidden_channels=64, out_channels=dataset.num_classes,
                  num_layers=num_layers)

# Initialize lazy modules.
with torch.no_grad():
    edge_index_dict = (data.adj_t_dict
                       if use_sparse_tensor else data.edge_index_dict)
    out = model(data.x_dict, edge_index_dict)

model_copy = copy.deepcopy(model)

kwargs = {'batch_size': 1024, 'num_workers': 0}
train_loader = NeighborLoader(data, num_neighbors=[10] * num_layers,
                              shuffle=True,
                              input_nodes=('paper', data['paper'].train_mask),
                              **kwargs)
val_loader = NeighborLoader(data, num_neighbors=[10] * num_layers,
                            shuffle=False,
                            input_nodes=('paper', data['paper'].val_mask),
                            **kwargs)


def train(loader, early_stop=None):
    model.train()
    model_copy.train()

    if early_stop is not None:
        it = 0
    for batch in tqdm(loader):

        edge_index_dict = (batch.adj_t_dict
                           if use_sparse_tensor else batch.edge_index_dict)
        batch_size = batch['paper'].batch_size
        batch_trim = copy.deepcopy(batch)
        edge_index_dict_trim = copy.deepcopy(edge_index_dict)

        if use_sparse_tensor:
            for key in edge_index_dict.keys():
                st = SparseTensor.from_storage(edge_index_dict[key].storage)
                st = copy.deepcopy(st)
                edge_index_dict[key] = st
            batch_trim.x_dict = batch.x_dict
            out_expected = model(batch.x_dict, edge_index_dict)[:batch_size]
        else:
            out_expected = model(batch.x_dict, edge_index_dict)[:batch_size]

        out_trim = model_copy(
            batch_trim.x_dict, edge_index_dict_trim,
            num_sampled_edges=batch_trim.num_sampled_edges_dict,
            num_sampled_nodes=batch_trim.num_sampled_nodes_dict,
            trim=True)[:batch_size]

        assert torch.allclose(out_expected, out_trim, atol=1e-6)

        loss_expected = F.cross_entropy(out_expected[:batch_size],
                                        batch['paper'].y[:batch_size])
        loss_expected.backward()
        loss_trim = F.cross_entropy(out_trim[:batch_size],
                                    batch_trim['paper'].y[:batch_size])
        loss_trim.backward()

        for expected, trimmed in zip(model.parameters(),
                                     model_copy.parameters()):
            assert torch.allclose(expected, trimmed, atol=1e-6)
        it += 1
        if early_stop == it:
            break


@torch.no_grad()
def test(loader, early_stop=None):
    model.eval()
    model_copy.eval()

    if early_stop is not None:
        it = 0
    for batch in tqdm(loader):

        edge_index_dict = (batch.adj_t_dict
                           if use_sparse_tensor else batch.edge_index_dict)
        batch_size = batch['paper'].batch_size
        batch_trim = copy.deepcopy(batch)
        edge_index_dict_trim = copy.deepcopy(edge_index_dict)

        if use_sparse_tensor:
            for key in edge_index_dict.keys():
                st = SparseTensor.from_storage(edge_index_dict[key].storage)
                st = copy.deepcopy(st)
                edge_index_dict[key] = st
            out_expected = model(batch.x_dict, edge_index_dict)[:batch_size]
        else:
            out_expected = model(batch.x_dict, edge_index_dict)[:batch_size]

        out_trim = model_copy(
            batch_trim.x_dict, edge_index_dict_trim,
            num_sampled_edges=batch_trim.num_sampled_edges_dict,
            num_sampled_nodes=batch_trim.num_sampled_nodes_dict,
            trim=True)[:batch_size]

        assert torch.allclose(out_expected, out_trim, atol=1e-6)

        it += 1
        if early_stop == it:
            break


early_stop = 10
train(train_loader, early_stop)
test(val_loader, early_stop)
