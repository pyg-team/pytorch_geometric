from torch.autograd import Variable
from copy import deepcopy
import torch_geometric
import torch
from torch_geometric import nn as nn_geom
from torch_geometric.data import Data
from torch import nn
from torch.utils.data import DataLoader
from torch_geometric.data import DataLoader, Dataset
import pandas as pd, numpy as np
import networkx as nx
from torch.autograd import Variable
from copy import deepcopy
from functools import reduce
import scipy.sparse as sps

def nx_graph_to_edgelist_tensor(G):
    edge_list=torch.tensor(np.array(list(G.edges())), dtype=torch.long)
    if 0 and not torch_geometric.utils.is_undirected(edge_list):
        edge_list=torch_geometric.utils.to_undirected(edge_list)
    if not torch_geometric.utils.contains_self_loops(edge_list): #0 and
        edge_list=torch_geometric.utils.add_self_loops(edge_list)
    return edge_list

def nx_graph2matrix(G):
    return nx.to_scipy_sparse_matrix(G).todense()

def X_df_to_tensor(x_df):
    return torch.tensor(x_df.values, dtype=torch.float)

def X_to_tensor(x):
    return torch.tensor(x, dtype=torch.float)

def graph_to_data(graph_dict, y=None):
    A=torch.tensor(nx_graph2matrix(graph_dict['A']))
    print(A)
    index, value=torch_geometric.utils.dense_to_sparse(A)
    if 0 and not torch_geometric.utils.contains_self_loops(index): #0 and
        edge_list=torch_geometric.utils.add_self_loops(index)
    return Data(x=X_to_tensor(graph_dict['X']), edge_index=index, edge_attr=value, y=y)

def return_adjacency(geom_data):
    edge_index, edge_attr = geom_data.edge_index, geom_data.edge_attr
    return torch_geometric.utils.sparse_to_dense(edge_index, edge_attr)

def normalize_similarity(x,graphs):
    return x[2]/(0.5*graphs[x[0]].size(0)*graphs[x[1]].size(0))


class SingleGraph(Dataset):
    def __init__(self, graph_dict, outcome_col=None): # if None do all vs all
        """graph_dict=dict(A=nx.Graph(...),X=pd.DataFrame(..., index=[node labels], columns=[covariates]))"""
        self.graph = graph_dict
        self.all_labels = np.array(list(self.graph['A'].nodes()))
        self.graph=graph_to_data(self.graph, outcome_col)
        self.length = len(self.all_labels)
        self.y = outcome_col
        if self.y==None:
            self.y = np.ones((self.length,))

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return self.length



def calc_loss(A_orig, A_new, d_real, d_fake, weights, norm):# weights = (float(A_orig.shape[0]**2-A_orig.sum())/A_orig.sum()).flatten()
    dc_real_loss=nn.BCELoss(reduction='mean')(d_real,torch.ones(d_real.size()))
    dc_fake_loss=nn.BCELoss(reduction='mean')(d_fake,torch.zeros(d_fake.size()))
    dc_gen_loss=nn.BCELoss(reduction='mean')(d_fake,torch.ones(d_fake.size()))
    #print(sum(A_new),sum(A_orig))
    A_loss = norm*nn.BCELoss(reduction='mean',weight=torch.tensor(weights,dtype=torch.float))(A_new,A_orig)
    #print(dc_real_loss.item(),dc_fake_loss.item(),A_loss.item())
    total_loss = dc_real_loss + dc_fake_loss + dc_gen_loss + A_loss
    return total_loss

def calc_weights(A):
    return (float(A.shape[0]**2-A.sum())/A.sum())#.flatten()

def calc_norm(A):
    return A.shape[0] * A.shape[0] / float((A.shape[0] * A.shape[0] - A.sum()) * 2)

class EncoderGCN(nn.Module):
    def __init__(self, n_total_features, n_latent, p_drop=0.):
        super(EncoderGCN, self).__init__()
        self.p_drop = p_drop
        self.n_total_features = n_total_features
        self.conv1 = nn_geom.GCNConv(self.n_total_features, 11)
        self.f1=nn.Sequential(nn.ReLU(),nn.Dropout(self.p_drop))
        self.conv2 = nn_geom.GCNConv(11, 11)
        self.f2=nn.Sequential(nn.ReLU(),nn.Dropout(self.p_drop))
        self.conv3 = nn_geom.GCNConv(11,n_latent)
        torch.nn.init.orthogonal_(self.conv1.weight)
        torch.nn.init.orthogonal_(self.conv2.weight)
        torch.nn.init.orthogonal_(self.conv3.weight)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.f1(self.conv1(x, edge_index)) # X,A
        #print(x)
        x = self.f2(self.conv2(x, edge_index))
        #print(x)
        x = self.conv3(x, edge_index)
        #print(x.mean())
        return x

class DecoderGCN(nn.Module):
    def __init__(self):
        super(DecoderGCN, self).__init__()
        self.output_layer = nn.Sigmoid()

    def forward(self, z):
        A=torch.mm(z,torch.t(z))
        #print(A)
        A=self.output_layer(A)
        return A

class Discriminator(nn.Module):
    def __init__(self, n_input):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(n_input,40), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(40,30), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(30,1),nn.Sigmoid())

    def forward(self, z):
        z=self.fc1(z)
        z=self.fc2(z)
        out=self.fc3(z)
        return out

class ARGA(nn.Module):
    def __init__(self, n_total_features, n_latent):
        super(ARGA, self).__init__()
        self.encoder = EncoderGCN(n_total_features, n_latent)
        self.decoder = DecoderGCN()
        self.discriminator = Discriminator(n_latent)

    def forward(self, x):
        z_fake = self.encoder(x)
        z_real=torch.randn(z_fake.size())
        #print(z_fake)
        A = self.decoder(z_fake)
        d_real=self.discriminator(z_real)
        d_fake=self.discriminator(z_fake)
        return A, d_real, d_fake

    def simulate(self, x):
        z_fake = self.encoder(x)
        A = self.decoder(z_fake)
        return A

class ARGA_Model:
    def __init__(self, optimizer, n_epochs, model=ARGA(20,30), n_nodes=20, n_latent=30, weights=None, norm=1., scheduler_opts=dict(scheduler='warm_restarts',lr_scheduler_decay=0.5,T_max=10,eta_min=5e-8,T_mult=2)):
        self.loss_fn = lambda A_orig, A_new, d_real, d_fake: calc_loss(A_orig, A_new, d_real, d_fake, weights, norm)
        self.model=model#(n_nodes,n_latent)
        self.optimizer = optimizer
        self.scheduler = Scheduler(optimizer, scheduler_opts)
        self.n_epochs = n_epochs
        self.flatten = lambda x: torch.flatten(x)
        self.sigmoid_fn = nn.Sigmoid()

    def train(self, data): # one
        self.model.train()
        self.total_loss = []
        self.total_lr = []
        A_real = return_adjacency(data)
        A_real = self.flatten(A_real).float()
        self.best_model=None
        for epoch in range(self.n_epochs):
            A_fake, d_real, d_fake = self.model(data)
            A_fake = self.flatten(A_fake)
            loss = self.loss_fn(A_real, A_fake, d_real, d_fake)
            loss.backward()
            self.optimizer.step()
            self.total_lr.append(self.scheduler.get_lr())
            self.scheduler.step()
            loss_float=loss.item()
            #print(loss_float)
            self.total_loss.append(loss_float)
            if loss_float <= min(self.total_loss):
                self.best_model = deepcopy(self.model)
        self.model = self.best_model


    def simulate(self, data):
        self.model.eval()
        A=self.model.simulate(data)
        return A.detach().numpy()
