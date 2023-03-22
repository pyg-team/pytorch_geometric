"""
Code taken from ogb examples and adapted
"""

import torch
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import BondEncoder

from torch_geometric.nn import GINConv as PyGINConv
from torch_geometric.nn import (
    GlobalAttention,
    GraphConv,
    MessagePassing,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from torch_geometric.utils import degree


def subgraph_pool(h_node, batched_data, pool):
    # Represent each subgraph as the pool of its node representations
    num_subgraphs = batched_data.num_subgraphs
    tmp = torch.cat([
        torch.zeros(1, device=num_subgraphs.device, dtype=num_subgraphs.dtype),
        torch.cumsum(num_subgraphs, dim=0)
    ])
    graph_offset = tmp[batched_data.batch]
    subgraph_id = batched_data.subgraph_batch + graph_offset

    return pool(h_node, subgraph_id)


### GIN convolution along the graph structure
class GINConv(MessagePassing):
    def __init__(self, in_dim, emb_dim):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GINConv, self).__init__(aggr="add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(in_dim, 2 * emb_dim),
                                       torch.nn.BatchNorm1d(2 * emb_dim),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(2 * emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        self.bond_encoder = BondEncoder(emb_dim=in_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)
        out = self.mlp(
            (1 + self.eps) * x +
            self.propagate(edge_index, x=x, edge_attr=edge_embedding))

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class ZINCGINConv(MessagePassing):
    def __init__(self, in_dim, emb_dim):
        super(ZINCGINConv, self).__init__(aggr="add")

        self.mlp = torch.nn.Sequential(torch.nn.Linear(in_dim, emb_dim),
                                       torch.nn.BatchNorm1d(emb_dim),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        self.bond_encoder = torch.nn.Embedding(4, in_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr.squeeze())
        out = self.mlp(
            (1 + self.eps) * x +
            self.propagate(edge_index, x=x, edge_attr=edge_embedding))

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class OriginalGINConv(torch.nn.Module):
    def __init__(self, in_dim, emb_dim):
        super(OriginalGINConv, self).__init__()
        mlp = torch.nn.Sequential(torch.nn.Linear(in_dim, emb_dim),
                                  torch.nn.BatchNorm1d(emb_dim),
                                  torch.nn.ReLU(),
                                  torch.nn.Linear(emb_dim, emb_dim))
        self.layer = PyGINConv(nn=mlp, train_eps=False)

    def forward(self, x, edge_index, edge_attr):
        return self.layer(x, edge_index)


### GCN convolution along the graph structure
class GCNConv(MessagePassing):
    def __init__(self, in_dim, emb_dim):
        super(GCNConv, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(in_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        self.bond_encoder = BondEncoder(emb_dim=emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.bond_encoder(edge_attr)

        row, col = edge_index

        deg = degree(row, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr=edge_embedding, norm=norm) + \
               F.relu(x + self.root_emb.weight) * 1. / deg.view(-1, 1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


### GNN to generate node embedding
class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layer, in_dim, emb_dim, drop_ratio=0.5, JK="last",
                 residual=False, GNNConv=GraphConv, num_random_features=0,
                 feature_encoder=lambda x: x):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers

        '''

        super(GNN_node, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual
        self.GNNConv = GNNConv

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = feature_encoder

        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layer):
            self.convs.append(
                GNNConv(emb_dim if layer != 0 else in_dim, emb_dim))
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        ### computing input node embedding
        h_list = [self.atom_encoder(x)]

        for layer in range(self.num_layer):
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            h = F.relu(h)

            if self.drop_ratio > 0.:
                h = F.dropout(h, self.drop_ratio, training=self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]
        elif self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)

        return node_representation


class GNN(torch.nn.Module):
    def __init__(self, num_tasks, num_layer=5, in_dim=300, emb_dim=300,
                 GNNConv=GraphConv, residual=False, drop_ratio=0.5, JK="last",
                 graph_pooling="mean", feature_encoder=lambda x: x):

        super(GNN, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.out_dim = self.emb_dim if self.JK == 'last' else self.emb_dim * self.num_layer + in_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        self.gnn_node = GNN_node(num_layer, in_dim, emb_dim, JK=JK,
                                 drop_ratio=drop_ratio, residual=residual,
                                 GNNConv=GNNConv,
                                 feature_encoder=feature_encoder)

        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn=torch.nn.Sequential(
                torch.nn.Linear(emb_dim, 2 *
                                emb_dim), torch.nn.BatchNorm1d(2 * emb_dim),
                torch.nn.ReLU(), torch.nn.Linear(2 * emb_dim, 1)))
        else:
            raise ValueError("Invalid graph pooling type.")

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)

        return subgraph_pool(h_node, batched_data, self.pool)


class GNNComplete(GNN):
    def __init__(self, num_tasks, num_layer=5, in_dim=300, emb_dim=300,
                 gnn_type='gin', residual=False, drop_ratio=0.5, JK="last",
                 graph_pooling="mean", feature_encoder=lambda x: x):

        if gnn_type == 'gin':
            GNNConv = GINConv
        elif gnn_type == 'originalgin':
            GNNConv = OriginalGINConv
        elif gnn_type == 'graphconv':
            GNNConv = GraphConv
        elif gnn_type == 'gcn':
            GNNConv = GCNConv
        elif gnn_type == 'zincgin':
            GNNConv = ZINCGINConv
        else:
            raise ValueError('Undefined GNN type called {}'.format(
                args.gnn_type))

        super(GNNComplete,
              self).__init__(num_tasks, num_layer, in_dim, emb_dim, GNNConv,
                             residual, drop_ratio, JK, graph_pooling,
                             feature_encoder)

        if gnn_type == 'graphconv':
            self.final_layers = torch.nn.Sequential(
                torch.nn.Linear(in_features=self.out_dim,
                                out_features=self.out_dim), torch.nn.ELU(),
                torch.nn.Linear(in_features=self.out_dim,
                                out_features=self.out_dim // 2),
                torch.nn.ELU(),
                torch.nn.Linear(in_features=self.out_dim // 2,
                                out_features=num_tasks))
        else:
            self.final_layers = torch.nn.Sequential(
                torch.nn.Linear(in_features=self.out_dim,
                                out_features=num_tasks), )

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)
        h_graph = self.pool(h_node, batched_data.batch)

        return self.final_layers(h_graph)


class ZincAtomEncoder(torch.nn.Module):
    def __init__(self, policy, emb_dim):
        super(ZincAtomEncoder, self).__init__()
        self.policy = policy
        self.num_added = 2
        self.enc = torch.nn.Embedding(21, emb_dim)

    def forward(self, x):
        if self.policy == 'ego_plus':
            return torch.hstack((x[:, :self.num_added],
                                 self.enc(x[:, self.num_added:].squeeze())))
        else:
            return self.enc(x.squeeze())
