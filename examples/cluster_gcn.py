import os
import time
import math    
import torch
import numpy as np
import torch.nn as nn
import os.path as osp
import torch.nn.functional as F
from sklearn.metrics import f1_score 
import torch_geometric.transforms as T
from torch_geometric.data import Batch
from torch_geometric.nn import ChebConv
from torch_geometric.datasets import PPI
from torch_geometric.datasets import Planetoid
from cluster import ClusterData, ClusterLoader
from torch_geometric.data import DataLoader, Data


path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'PPI')
train_dataset = PPI(path, split='train')    #20graphs
val_dataset = PPI(path, split='val')        #2graphs
test_dataset = PPI(path, split='test')      #2graphs
dataset = PPI(path)


train_data_list = [data for data in train_dataset]
for data in train_data_list:
    data.train_mask = torch.ones(data.num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    
val_data_list = [data for data in val_dataset]
for data in val_data_list:
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.val_mask = torch.ones(data.num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

test_data_list = [data for data in test_dataset]
for data in test_data_list:
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask = torch.ones(data.num_nodes, dtype=torch.bool)


data = Batch.from_data_list(train_data_list + val_data_list + test_data_list)
cluster_data = ClusterData(data, num_parts=50, recursive=False,
                            save_dir=dataset.processed_dir)
loader = ClusterLoader(cluster_data, batch_size=1, shuffle=True,
                        num_workers=0)    



class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Net, self).__init__()
        dim = 2048
        self.gcn1 = ChebConv(in_channels, dim, K=1)
        self.lin1 = nn.Linear(in_channels, dim)
        
        self.gcn2 = ChebConv(dim, dim, K=1)
        self.lin2 = nn.Linear(dim, dim)
        
        self.gcn3 = ChebConv(dim, dim, K=1)
        self.lin3 = nn.Linear(dim, dim)
        
        self.gcn4 = ChebConv(dim, dim, K=1)
        self.lin4 = nn.Linear(dim, dim)
        
        self.gcn5 = ChebConv(dim, out_channels, K=1)
        self.lin5 = nn.Linear(dim, out_channels)
        

    def forward(self, x, edge_index):
        x = F.elu(self.gcn1(x, edge_index) + self.lin1(x))
        x = F.elu(self.gcn2(x, edge_index) + self.lin2(x))
        x = F.elu(self.gcn3(x, edge_index) + self.lin3(x))
        x = F.elu(self.gcn4(x, edge_index) + self.lin4(x))
        x = self.gcn5(x, edge_index) + self.lin5(x)

        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(dataset.num_features, dataset.num_classes).to(device)
loss_op = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)


def train():
    model.train()
    total_loss = total_nodes = 0
    for data in loader:
        #because some data.train_mask in loader are all False, it would cause the loss=nan
        if data.train_mask.sum().item() == 0:
            continue
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = loss_op(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()        
        nodes = data.train_mask.sum().item()
        total_loss += loss.item() * nodes
        total_nodes += nodes

    return total_loss / total_nodes     
        
        
def test():
    model.eval()
    ys, preds = [[], [], []], [[], [], []]
    accs = []
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data.x, data.edge_index)
        masks = [data.train_mask, data.val_mask, data.test_mask]
        for i, mask in enumerate(masks):
            ys[i].append(data.y[mask])
            preds[i].append((out[mask]>0).float().cpu())
            
    for i in range(3):
        y, pred = torch.cat(ys[i], dim=0).cpu(), torch.cat(preds[i], dim=0).cpu()
        acc = f1_score(y, pred, average='micro') if pred.sum() > 0 else 0
        accs.append(acc)
    return accs
        
       
for epoch in range(1, 500):
    loss = train()     
    accs= test()
    print('Epoch: {:02d}, Loss: {:.4f}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}, '
          .format(epoch, loss, *accs))
        
        
        
    
