#!/usr/bin/env python
# coding: utf-8

# # dgl mixhop on cora

# In[3]:


import copy
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import dgl
import dgl.function as fn

from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset,PPIDataset
from tqdm import trange


# ## mixhop dgl version

# In[28]:


class MixHopConvDGL(nn.Module):

    def __init__(self,
                 in_dim,
                 out_dim,
                 p=[0, 1, 2],
                 dropout=0,
                 activation=None,
                 batchnorm=False):
        super(MixHopConvDGL, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.p = p
        self.activation = activation
        self.batchnorm = batchnorm

        # define dropout layer
        self.dropout = nn.Dropout(dropout)

        # define batch norm layer
        if self.batchnorm:
            self.bn = nn.BatchNorm1d(out_dim * len(p))
        
        # define weight dict for each power j
        self.weights = nn.ModuleDict({
            str(j): nn.Linear(in_dim, out_dim, bias=False) for j in p
        })

    def forward(self, graph, feats):
        with graph.local_scope():
            # assume that the graphs are undirected and graph.in_degrees() is the same as graph.out_degrees()
            degs = graph.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5).to(feats.device).unsqueeze(1) #对graph做标准化

            max_j = max(self.p) + 1
            outputs = []

            for j in range(max_j):

                if j in self.p:
                    output = self.weights[str(j)](feats)
                    outputs.append(output)

                feats = feats * norm
                graph.ndata['h'] = feats

                graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h')) # dgl的update all

                # 第一个function为message passing function 对应 pyg 中的message function
                # 第二个function 为reduce function，即聚合过程，对应pyg中的super().__init__(aggr='add') 的聚合

                feats = graph.ndata.pop('h')
                feats = feats * norm
            
            final = torch.cat(outputs, dim=1)
            
            if self.batchnorm:
                final = self.bn(final)
            
            if self.activation is not None:
                final = self.activation(final)
            
            final = self.dropout(final)

            return final


# In[31]:




class MixHopDGL(nn.Module):
    def __init__(self,
                 in_dim,
                 hid_dim, 
                 out_dim,
                 num_layers=2,
                 p=[0, 1, 2],
                 input_dropout=0.0,
                 layer_dropout=0.0,
                 activation=None,
                 batchnorm=False):
        super(MixHopDGL, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.p = p
        self.input_dropout = input_dropout
        self.layer_dropout = layer_dropout
        self.activation = activation
        self.batchnorm = batchnorm

        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(self.input_dropout)

        # Input layer
        self.layers.append(MixHopConvDGL(self.in_dim,
                                      self.hid_dim,
                                      p=self.p,
                                      dropout=self.input_dropout,
                                      activation=self.activation,
                                      batchnorm=self.batchnorm))
        
        # Hidden layers with n - 1 MixHopConv layers
        for i in range(self.num_layers - 2):
            self.layers.append(MixHopConvDGL(self.hid_dim * len(p),
                                          self.hid_dim,
                                          p=self.p,
                                          dropout=self.layer_dropout,
                                          activation=self.activation,
                                          batchnorm=self.batchnorm))
        
        self.fc_layers = nn.Linear(self.hid_dim * len(p), self.out_dim, bias=False)

    def forward(self, graph, feats):
        feats = self.dropout(feats)
        for layer in self.layers:
            feats = layer(graph, feats)
        
        feats = self.fc_layers(feats)

        return feats


# ## mixhop pyg version

# In[32]:


import torch
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class MixHopConv(MessagePassing):
    def __init__(self,
                 in_dim,
                 out_dim,
                 p=[0, 1, 2],
                 dropout=0,
                 activation=None,
                 batchnorm=False):
      
        super().__init__(aggr='add')  

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.p = p
        self.activation = activation
        self.batchnorm = batchnorm

        # define dropout layer
        self.dropout = nn.Dropout(dropout)

        # define batch norm layer
        if self.batchnorm:
            self.bn = nn.BatchNorm1d(out_dim * len(p))
        
        # define weight dict for each power j
        self.weights = nn.ModuleDict({
            str(j): nn.Linear(in_dim, out_dim, bias=False) for j in p
        })

        self.max_j =  max(self.p) + 1




    def forward(self, x, edge_index):

        outputs = []

        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        feats = x


        for j in range(self.max_j):

            if j in self.p:
                output = self.weights[str(j)](feats)
                outputs.append(output)

            
            feats = self.propagate(edge_index, x=x, norm=norm)

        final = torch.cat(outputs, dim=1)

        if self.batchnorm:
            final = self.bn(final)
        
        if self.activation is not None:
            final = self.activation(final)
        
        final = self.dropout(final)

        return final

    def message(self, x_j, norm):

        return norm.view(-1, 1) * x_j


# In[36]:




class MixHop(nn.Module):
    def __init__(self,
                 in_dim,
                 hid_dim, 
                 out_dim,
                 num_layers=2,
                 p=[0, 1, 2],
                 input_dropout=0.0,
                 layer_dropout=0.0,
                 activation=None,
                 batchnorm=False):
        super(MixHop, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.p = p
        self.input_dropout = input_dropout
        self.layer_dropout = layer_dropout
        self.activation = activation
        self.batchnorm = batchnorm

        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(self.input_dropout)

        # Input layer
        self.layers.append(MixHopConv(self.in_dim,
                                      self.hid_dim,
                                      p=self.p,
                                      dropout=self.input_dropout,
                                      activation=self.activation,
                                      batchnorm=self.batchnorm))
        
        # Hidden layers with n - 1 MixHopConv layers
        for i in range(self.num_layers - 2):
            self.layers.append(MixHopConv(self.hid_dim * len(p),
                                          self.hid_dim,
                                          p=self.p,
                                          dropout=self.layer_dropout,
                                          activation=self.activation,
                                          batchnorm=self.batchnorm))
        
        self.fc_layers = nn.Linear(self.hid_dim * len(p), self.out_dim, bias=False)

    def forward(self, x,edge_index):
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, edge_index)
        
        x = self.fc_layers(x)

        return x


# ## cora dgl 

# In[59]:


dataset = CoraGraphDataset()
graph = dataset[0]
graph = dgl.add_self_loop(graph)
device = 'cpu'


# In[60]:


n_classes = dataset.num_classes

labels = graph.ndata.pop('label').to(device).long()

feats = graph.ndata.pop('feat').to(device)
n_features = feats.shape[-1]

train_mask = graph.ndata.pop('train_mask')
val_mask = graph.ndata.pop('val_mask')
test_mask = graph.ndata.pop('test_mask')

train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze().to(device)
val_idx = torch.nonzero(val_mask, as_tuple=False).squeeze().to(device)
test_idx = torch.nonzero(test_mask, as_tuple=False).squeeze().to(device)

graph = graph.to(device)


# In[35]:




model = MixHopDGL(in_dim=n_features,
                hid_dim=100,
                out_dim=n_classes,
                num_layers=3,
                p=[0, 1, 2],
                input_dropout=0.6,
                layer_dropout=0.0,
                activation=torch.tanh,
                batchnorm=True)

model = model.to(device)
best_model = copy.deepcopy(model)

loss_fn = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-3)
scheduler = optim.lr_scheduler.StepLR(opt, 40, gamma=0.01)


acc = 0
no_improvement = 0
epochs = trange(1000, desc='Accuracy & Loss')
early_stopping = 25

for _ in epochs:
    # Training using a full graph
    model.train()

    logits = model(feats, edge_index)

    # compute loss
    train_loss = loss_fn(logits[train_idx], labels[train_idx])
    train_acc = torch.sum(logits[train_idx].argmax(dim=1) == labels[train_idx]).item() / len(train_idx)

    # backward
    opt.zero_grad()
    train_loss.backward()
    opt.step()

    # Validation using a full graph
    model.eval()

    with torch.no_grad():
        valid_loss = loss_fn(logits[val_idx], labels[val_idx])
        valid_acc = torch.sum(logits[val_idx].argmax(dim=1) == labels[val_idx]).item() / len(val_idx)

    # Print out performance
    epochs.set_description('Train Acc {:.4f} | Train Loss {:.4f} | Val Acc {:.4f} | Val loss {:.4f}'.format(
        train_acc, train_loss.item(), valid_acc, valid_loss.item()))
    
    if valid_acc < acc:
        no_improvement += 1
        if no_improvement == early_stopping:
            print('Early stop.')
            break
    else:
        no_improvement = 0
        acc = valid_acc
        best_model = copy.deepcopy(model)
    
    scheduler.step()

best_model.eval()
logits = best_model(graph, feats)
test_acc = torch.sum(logits[test_idx].argmax(dim=1) == labels[test_idx]).item() / len(test_idx)

print("Test Acc {:.4f}".format(test_acc))


# ## cora pyg

# In[45]:


dataset = CoraGraphDataset()
graph = dataset[0]
graph = dgl.add_self_loop(graph)
device = 'cpu'

n_classes = dataset.num_classes

labels = graph.ndata.pop('label').to(device).long()

feats = graph.ndata.pop('feat').to(device)
n_features = feats.shape[-1]

train_mask = graph.ndata.pop('train_mask')
val_mask = graph.ndata.pop('val_mask')
test_mask = graph.ndata.pop('test_mask')

train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze().to(device)
val_idx = torch.nonzero(val_mask, as_tuple=False).squeeze().to(device)
test_idx = torch.nonzero(test_mask, as_tuple=False).squeeze().to(device)

graph = graph.to(device)


# In[61]:


edge_index = torch.vstack(graph.edges()).contiguous()
edge_index


# In[66]:


model = MixHop(in_dim=n_features,
                hid_dim=100,
                out_dim=n_classes,
                num_layers=3,
                p=[0, 1, 2],
                input_dropout=0.6,
                layer_dropout=0.0,
                activation=torch.tanh,
                batchnorm=True)


# In[67]:


model = model.to(device)
best_model = copy.deepcopy(model)

loss_fn = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-3)
scheduler = optim.lr_scheduler.StepLR(opt, 40, gamma=0.01)


# In[68]:


acc = 0
no_improvement = 0
epochs = trange(1000, desc='Accuracy & Loss')
early_stopping = 25


# In[69]:


for _ in epochs:
    # Training using a full graph
    model.train()

    logits = model(feats, edge_index)

    # compute loss
    train_loss = loss_fn(logits[train_idx], labels[train_idx])
    train_acc = torch.sum(logits[train_idx].argmax(dim=1) == labels[train_idx]).item() / len(train_idx)

    # backward
    opt.zero_grad()
    train_loss.backward()
    opt.step()

    # Validation using a full graph
    model.eval()

    with torch.no_grad():
        valid_loss = loss_fn(logits[val_idx], labels[val_idx])
        valid_acc = torch.sum(logits[val_idx].argmax(dim=1) == labels[val_idx]).item() / len(val_idx)

    # Print out performance
    epochs.set_description('Train Acc {:.4f} | Train Loss {:.4f} | Val Acc {:.4f} | Val loss {:.4f}'.format(
        train_acc, train_loss.item(), valid_acc, valid_loss.item()))
    
    if valid_acc < acc:
        no_improvement += 1
        if no_improvement == early_stopping:
            print('Early stop.')
            break
    else:
        no_improvement = 0
        acc = valid_acc
        best_model = copy.deepcopy(model)
    
    scheduler.step()

best_model.eval()
logits = best_model(feats, edge_index)
test_acc = torch.sum(logits[test_idx].argmax(dim=1) == labels[test_idx]).item() / len(test_idx)

print("Test Acc {:.4f}".format(test_acc))

