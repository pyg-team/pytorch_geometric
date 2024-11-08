import os.path as osp
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.data import Data
from torch_geometric.contrib.nn import NodeMixup

# Function to get the shuffled version of the input graph
def shuffle_data(data: Data):
    data = copy.deepcopy(data)
    id_new_value_old = np.arange(data.num_nodes)
    train_id_shuffle = copy.deepcopy(data.train_id)
    np.random.shuffle(train_id_shuffle)
    id_new_value_old[data.train_id] = train_id_shuffle

    row, col = data.edge_index[0], data.edge_index[1]
    id_old_value_new = torch.zeros(id_new_value_old.shape[0], dtype=torch.long)
    id_old_value_new[id_new_value_old] = torch.arange(id_new_value_old.shape[0], dtype=torch.long)
    row, col = id_old_value_new[row], id_old_value_new[col]
    data.edge_index = torch.stack([row, col], dim=0)
    
    return data, id_new_value_old

# Load dataset
dataset_name = 'Pubmed'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset_name)
dataset = Planetoid(path, dataset_name, transform=NormalizeFeatures())
data = dataset[0]

# Split dataset
node_id = np.arange(data.num_nodes)
np.random.shuffle(node_id)
data.train_id = node_id[:int(data.num_nodes * 0.6)]
data.val_id = node_id[int(data.num_nodes * 0.6):int(data.num_nodes * 0.8)]
data.test_id = node_id[int(data.num_nodes * 0.8):]

# Initialize the model
num_layers = 3
in_channels = dataset.num_node_features
hidden_channels = 256
out_channels = dataset.num_classes
model = NodeMixup(
    num_layers=num_layers,
    in_channels=in_channels,
    hidden_channels=hidden_channels,
    out_channels=out_channels,
    dropout=0.4
)

# Define optimizer and device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training function
def train(data):
    model.train()
    lam = np.random.beta(4.0, 4.0)

    data_b, id_new_value_old = shuffle_data(data)
    data_b = data_b.to(device)

    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data_b.edge_index, lam, id_new_value_old)
    loss = F.nll_loss(out[data.train_id], data.y[data.train_id]) * lam + \
           F.nll_loss(out[data.train_id], data_b.y[data.train_id]) * (1 - lam)
    loss.backward()
    optimizer.step()
    
    return loss.item()

# Testing function
@torch.no_grad()
def test(data):
    model.eval()
    out = model(data.x, data.edge_index, data.edge_index, 1, np.arange(data.num_nodes))
    pred = out.argmax(dim=-1)
    correct = pred.eq(data.y)
    accs = []
    for _, id_ in data('train_id', 'val_id', 'test_id'):
        accs.append(correct[id_].sum().item() / id_.shape[0])
    return accs

# Training loop
best_acc = 0
for epoch in range(1, 301):
    loss = train(data)
    accs = test(data)
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train Acc: {accs[0]:.4f}, Test Acc: {accs[2]:.4f}')