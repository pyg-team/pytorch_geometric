import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import DBLP
from torch_geometric.nn import SAGEConv, HeteroConv, Linear

path = osp.join(osp.dirname(osp.realpath(__file__)), '../../data/DBLP')
dataset = DBLP(path)
data = dataset[0]

hidden_channels = 64
out_channels = 4
num_layers = 2
print(data)

# We initialize conference node features with a single feature.
data['conference'].x = torch.ones(data['conference'].num_nodes, 1)

def build_message_convs(data, hidden_channels, out_channels, num_layers):
    for node_type in data.node_types:
        print(data[node_type].x.size())
        
    convs = []
    #input layer
    input_layer_convs = {}
    for message_type in data.edge_index_dict:
        node_type1, e_type, node_type2 = message_type
        input_layer_convs[message_type] = SAGEConv(
            in_channels=(data[node_type1].x.size(1), data[node_type2].x.size(1)), out_channels=hidden_channels)
    convs.append(input_layer_convs)
        
    for _ in range(num_layers):
        hidden_layer_convs = {}
        for message_type in data.edge_index_dict:
            node_type1, e_type, node_type2 = message_type
            hidden_layer_convs[message_type] = SAGEConv(in_channels=hidden_channels, out_channels=hidden_channels)
        convs.append(hidden_layer_convs)
        
    return convs
    

class HeteroSageGNN(torch.nn.Module):
    def __init__(self, data, hidden_channels, out_channels, num_layers):
        super().__init__()
        
        convs = build_message_convs(data, hidden_channels, out_channels, num_layers)
        self.convs = nn.ModuleList([HeteroConv(conv) for conv in convs])
        
        self.relus1 = nn.ModuleDict()
        for node_type in data.x_dict:
            self.relus1[node_type] = nn.LeakyReLU()

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.convs[0](x_dict, edge_index_dict)
        x_dict = HeteroConv.forward_op(x_dict, self.relus1)
        x_dict = self.convs[1](x_dict, edge_index_dict)
        
        return self.lin(x_dict['author'])


model = HeteroSageGNN(data, hidden_channels, out_channels, num_layers)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data, model = data.to(device), model.to(device)

with torch.no_grad():  # Initialize lazy modules.
    out = model(data.x_dict, data.edge_index_dict)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    mask = data['author'].train_mask
    loss = F.cross_entropy(out[mask], data['author'].y[mask])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test():
    model.eval()
    pred = model(data.x_dict, data.edge_index_dict).argmax(dim=-1)

    accs = []
    for split in ['train_mask', 'val_mask', 'test_mask']:
        mask = data['author'][split]
        acc = (pred[mask] == data['author'].y[mask]).sum() / mask.sum()
        accs.append(float(acc))
    return accs


for epoch in range(1, 101):
    loss = train()
    train_acc, val_acc, test_acc = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
          f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')