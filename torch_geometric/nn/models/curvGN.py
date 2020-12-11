import torch
import torch.nn.functional as F
import numpy as np
from torch.nn import Sequential as seq, LeakyReLU, Linear
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_geometric.data import download_url
import networkx as nx
import os


class ConvCurv(torch.nn.Module):
    r"""The Curvature Graph Network model from the `"Curvature Graph Network"
       <https://openreview.net/forum?id=BylEqnVFDB>` paper which adds
       ricci curvature as topological features to GNN model,
       and achieves state-of-the-art result

       Args:
           num_features (int): Size of input node features.
           num_classes (int): Number of input node labels.
           w_mul (array): Ricci curvature of the graph,
           can be computed through function 'compute_ricci_curvature'.
           data (torch_geometric.data): Input graph data.
       """

    def __init__(self, num_features, num_classes, w_mul):
        super(ConvCurv, self).__init__()
        self.conv1 = curvGN(num_features, 64, w_mul)
        self.conv2 = curvGN(64, num_classes, w_mul)

    def forward(self, data):
        x = F.dropout(data.x, p=0.8, training=self.training)
        x = self.conv1(x, data.edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.8, training=self.training)
        x = self.conv2(x, data.edge_index)
        return F.log_softmax(x, dim=1)


class curvGN(MessagePassing):
    def __init__(self, in_channels, out_channels, w_mul):
        super(curvGN, self).__init__(aggr='add')   # "Add" aggregation.
        self.w_mul = w_mul
        self.lin = Linear(in_channels, out_channels)
        widths = [1, out_channels]
        self.w_mlp_out = create_wmlp(widths, out_channels, 1)

    def forward(self, x, edge_index):
        x = self.lin(x)
        out_weight = self.w_mlp_out(self.w_mul)
        out_weight = softmax(out_weight, edge_index[0])
        return self.propagate(x=x, edge_index=edge_index,
                              out_weight=out_weight)

    def message(self, x_j, edge_index, out_weight):
        return out_weight * x_j

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]
        # Return new node embeddings.
        return aggr_out


def create_wmlp(widths, nfeato, lbias):
    mlp_modules = []
    for k in range(len(widths) - 1):
        mlp_modules.append(Linear(widths[k], widths[k + 1], bias=False))
        mlp_modules.append(LeakyReLU(0.2, True))
    mlp_modules.append(Linear(widths[len(widths) - 1], nfeato, bias=lbias))
    return seq(*mlp_modules)


def compute_ricci_curvature(dataset, d_name, mode='download'):

    r"""The function to compute ricci curvature for the input data,
    download the computed ricci curvature data on
    pkuyzy's github as default setting

       Args:
           dataset (torch_geometric.datasets): Input graph dataset
           (can use torch_geometric.datasets to load such dataset ).
           d_name (string): Name of input data name.
           mode (string): Whether to download ricci curvature data,
           or compute it using the code
       """

    if not os.path.exists('./data'):
        os.mkdir('./data')
    if not os.path.exists('./data/curvature'):
        os.mkdir('./data/curvature')
    filename = './data/curvature/graph_' + d_name + '.edge_list'
    if os.path.exists(filename):
        print("curvature file exists, directly loading")
        ricci_list = load_ricci_file(filename)
    else:
        if mode == 'download':
            curv_url = 'https://raw.githubusercontent.com/pkuyzy/curv_/main'
            download_url('{}/graph_{}.edge_list'.format(curv_url, d_name),
                         './data/curvature/')
            ricci_list = load_ricci_file(filename)
        else:
            from GraphRicciCurvature.OllivierRicci import OllivierRicci
            print("start writing ricci curvature")
            Gd = nx.Graph()
            ricci_edge_index_ = np.array(dataset[0].edge_index)
            ricci_edge_index = [(ricci_edge_index_[0, i],
                                 ricci_edge_index_[1, i]) for i in
                                range(np.shape(dataset[0].edge_index)[1])]
            Gd.add_edges_from(ricci_edge_index)
            Gd_OT = OllivierRicci(Gd, alpha=0.5, method="OTD", verbose="INFO")
            print("adding edges finished")
            Gd_OT.compute_ricci_curvature()
            ricci_list = []
            for n1, n2 in Gd_OT.G.edges():
                ricci_list.append([n1, n2, Gd_OT.G[n1][n2]['ricciCurvature']])
                ricci_list.append([n2, n1, Gd_OT.G[n1][n2]['ricciCurvature']])
            ricci_list = sorted(ricci_list)
            ricci_file = open(filename, 'w')
            for ricci_i in range(len(ricci_list)):
                ricci_file.write(
                    str(ricci_list[ricci_i][0]) + " " +
                    str(ricci_list[ricci_i][1]) + " " +
                    str(ricci_list[ricci_i][2]) + "\n")
            ricci_file.close()
            print("computing ricci curvature finished")
    return ricci_list


def num(strings):
    try:
        return int(strings)
    except ValueError:
        return float(strings)


def load_ricci_file(filename):
    if os.path.exists(filename):
        f = open(filename)
        cur_list = list(f)
        ricci_list = [[] for i in range(len(cur_list))]
        for i in range(len(cur_list)):
            ricci_list[i] = [num(s) for s in cur_list[i].split(' ', 2)]
        ricci_list = sorted(ricci_list)
        return ricci_list
    else:
        print("Error: no curvature files found")
