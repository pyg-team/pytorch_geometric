import os.path as osp
import math
from shutil import rmtree

import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.sparse.csgraph import shortest_path
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Linear, Conv1d, ReLU

from torch_geometric.utils import (negative_sampling, remove_self_loops,
                                   add_self_loops, train_test_split_edges, 
                                   k_hop_subgraph, to_scipy_sparse_matrix)
from torch_geometric.data import Data, InMemoryDataset, DataLoader
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, global_sort_pool

torch.manual_seed(12345)

dataset = 'Cora'
num_hops = 2
use_attribute = False
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
data = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = data[0]

# Train/validation/test
data.train_mask = data.val_mask = data.test_mask = data.y = None
data = train_test_split_edges(data)


class SealDataset(InMemoryDataset):
    def __init__(self, data_list, root, transform=None, pre_transform=None):
        self.data_list = data_list
        super(SealDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        data_list = self.data_list

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        del self.data_list


class DGCNN(torch.nn.Module):
    # Deep Graph Convolutional Neural Network [Zhang et.al. AAAI 2018]
    def __init__(self, dataset, num_layers, hidden, gconv=GCNConv, k=0.6):
        super(DGCNN, self).__init__()
        if k < 1:  # transform percentile to number
            node_nums = sorted([g.num_nodes for g in dataset])
            k = node_nums[int(math.ceil(k * len(node_nums)))-1]
            k = max(10, k)  # no smaller than 10
        self.k = int(k)
        print('k used in sortpooling is:', self.k)

        self.convs = nn.ModuleList()
        self.convs.append(gconv(dataset.num_features, hidden))
        for i in range(0, num_layers-1):
            self.convs.append(gconv(hidden, hidden))
        self.convs.append(gconv(hidden, 1))  # add a 1-dim layer for SortPooling

        conv1d_channels = [16, 32]
        conv1d_activation = nn.ReLU()
        self.total_latent_dim = hidden * num_layers + 1
        conv1d_kws = [self.total_latent_dim, 5]
        self.conv1d_params1 = Conv1d(1, conv1d_channels[0], conv1d_kws[0], 
                                     conv1d_kws[0])
        self.maxpool1d = nn.MaxPool1d(2, 2)
        self.conv1d_params2 = Conv1d(conv1d_channels[0], conv1d_channels[1], 
                                     conv1d_kws[1], 1)
        dense_dim = int((self.k - 2) / 2 + 1)
        self.dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        self.lin1 = Linear(self.dense_dim, 128)
        self.lin2 = Linear(128, dataset.num_classes)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.conv1d_params1.reset_parameters()
        self.conv1d_params2.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        concat_states = []
        for conv in self.convs:
            x = torch.tanh(conv(x, edge_index))
            concat_states.append(x)
        concat_states = torch.cat(concat_states, 1)

        # global pooling
        x = global_sort_pool(concat_states, batch, self.k)
        x = x.unsqueeze(1)  # num_graphs * 1 * (k*hidden)
        x = F.relu(self.conv1d_params1(x))
        x = self.maxpool1d(x)
        x = F.relu(self.conv1d_params2(x))
        x = x.view(len(x), -1)  # num_graphs * dense_dim

        # MLP
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)


def extract_enclosing_subgraphs(target_edge_index, num_hops, x, edge_index, y):
    subgraphs = []
    node_labels_list = []
    for i in range(target_edge_index.shape[1]):
        src, dst = target_edge_index[0, i], target_edge_index[1, i]
        sub_nodes, sub_edge_index, _, _ = k_hop_subgraph(
            [src, dst], num_hops, edge_index, relabel_nodes=True)
        # remove target link from the subgraph
        src = sub_nodes.tolist().index(src)  # convert to new index
        dst = sub_nodes.tolist().index(dst)
        src_dst_mask = (sub_edge_index[0] == src) & (sub_edge_index[1] == dst) 
        dst_src_mask = (sub_edge_index[0] == dst) & (sub_edge_index[1] == src)
        target_link_mask = src_dst_mask | dst_src_mask
        sub_edge_index = sub_edge_index[:, ~target_link_mask]
        sub_x = x[sub_nodes]
        subgraph = Data(sub_x, sub_edge_index, None, y)
        subgraph.src_x = sub_x[src].view(1, -1)
        subgraph.dst_x = sub_x[dst].view(1, -1)
        # node labeling
        adj = to_scipy_sparse_matrix(sub_edge_index, num_nodes=len(sub_nodes))
        node_labels = drnl_node_labeling(adj.tocsc(), src, dst)
        node_labels_list.append(node_labels)
        subgraph.node_labels = node_labels
        subgraphs.append(subgraph)
    return subgraphs


def process_features(subgraphs, max_node_label, use_attribute=False):
    for subgraph in subgraphs:
        node_labels = torch.FloatTensor(one_hot(subgraph.node_labels, 
                                        max_node_label+1))
        if use_attribute:
            subgraph.x = torch.cat([node_labels, subgraph.x], 1)
        else:
            subgraph.x = node_labels
    return subgraphs
    

def drnl_node_labeling(subgraph, src, dst):
    # double-radius node labeling (DRNL)
    if src > dst:
        src, dst = dst, src
    K = subgraph.shape[0]
    nodes_wo_src = list(range(src)) + list(range(src+1, K))
    subgraph_wo_src = subgraph[nodes_wo_src, :][:, nodes_wo_src]
    nodes_wo_dst = list(range(dst)) + list(range(dst+1, K))
    subgraph_wo_dst = subgraph[nodes_wo_dst, :][:, nodes_wo_dst]
    dist_to_dst = shortest_path(subgraph_wo_src, directed=False, unweighted=True)
    dist_to_dst = dist_to_dst[:, dst-1]
    dist_to_dst = np.insert(dist_to_dst, src, 0, axis=0)
    dist_to_src = shortest_path(subgraph_wo_dst, directed=False, unweighted=True)
    dist_to_src = dist_to_src[:, src]
    dist_to_src = np.insert(dist_to_src, dst, 0, axis=0)
    d = (dist_to_src + dist_to_dst).astype(int)
    d_over_2, d_mod_2 = np.divmod(d, 2)
    labels = (
        1 + np.minimum(dist_to_src, dist_to_dst).astype(int) + 
        d_over_2 * (d_over_2 + d_mod_2 - 1)
    )
    labels[src] = 1
    labels[dst] = 1
    labels[np.isinf(labels)] = 0
    labels[labels>1e6] = 0  # set inf labels to 0
    labels[labels<-1e6] = 0  # set -inf labels to 0
    return labels


def one_hot(idx, length):
    idx = np.array(idx)
    x = np.zeros([len(idx), length])
    x[np.arange(len(idx)), idx] = 1.0
    return x


def train(loader):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = F.nll_loss(out, batch.y.view(-1))
        loss.backward()
        total_loss += loss.item() * batch.num_graphs
        optimizer.step()
    return loss


def test(loader):
    model.eval()
    link_probs = []
    link_labels = []
    for batch in loader:
        batch = batch.to(device)
        link_probs.append(torch.exp(model(batch)[:, 1]).detach())
        link_labels.append(batch.y.view(-1))

    link_labels = torch.cat(link_labels).cpu().numpy()
    link_probs = torch.cat(link_probs).cpu().numpy()
    auc = roc_auc_score(link_labels, link_probs)
    return auc


# Prepare negative training links
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x, pos_edge_index = data.x, data.train_pos_edge_index

_edge_index, _ = remove_self_loops(pos_edge_index)
pos_edge_index_with_self_loops, _ = add_self_loops(_edge_index,
                                                   num_nodes=x.size(0))

neg_edge_index = negative_sampling(
    edge_index=pos_edge_index_with_self_loops, num_nodes=x.size(0),
    num_neg_samples=pos_edge_index.size(1))


# Extract enclosing subgraphs
print("Extract enclosing subgraphs...")
pos_train_graphs = extract_enclosing_subgraphs(pos_edge_index, num_hops, x, 
                                               pos_edge_index, 1)
neg_train_graphs = extract_enclosing_subgraphs(neg_edge_index, num_hops, x, 
                                               pos_edge_index, 0)
train_graphs = pos_train_graphs + neg_train_graphs

pos_val_graphs = extract_enclosing_subgraphs(data.val_pos_edge_index, 
                                             num_hops, x, pos_edge_index, 1)
neg_val_graphs = extract_enclosing_subgraphs(data.val_neg_edge_index, 
                                             num_hops, x, pos_edge_index, 0)
val_graphs = pos_val_graphs + neg_val_graphs

pos_test_graphs = extract_enclosing_subgraphs(data.test_pos_edge_index, 
                                              num_hops, x, pos_edge_index, 1) 
neg_test_graphs = extract_enclosing_subgraphs(data.test_neg_edge_index, 
                                              num_hops, x, pos_edge_index, 0)
test_graphs = pos_test_graphs + neg_test_graphs


max_node_label = max(np.concatenate([g.node_labels 
                     for g in train_graphs + val_graphs + test_graphs]))
train_graphs = process_features(train_graphs, max_node_label, use_attribute)
train_path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 
                      dataset, 'seal_train')
if osp.isdir(train_path):
    rmtree(train_path)
train_dataset = SealDataset(train_graphs, train_path)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_graphs = process_features(val_graphs, max_node_label, use_attribute)
val_path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 
                    dataset, 'seal_val')
if osp.isdir(val_path):
    rmtree(val_path)
val_dataset = SealDataset(val_graphs, val_path)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

test_graphs = process_features(test_graphs, max_node_label, use_attribute)
test_path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 
                     dataset, 'seal_test')
if osp.isdir(test_path):
    rmtree(test_path)
test_dataset = SealDataset(test_graphs, test_path)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)


# Train and test
model = DGCNN(train_dataset, 3, 32).to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)

best_val_auc = test_auc = 0
for epoch in range(1, 51):
    train_loss = train(train_loader)
    val_auc = test(val_loader)
    temp_test_auc = test(test_loader)
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        test_auc = temp_test_auc
    log = 'Epoch: {:03d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, train_loss, best_val_auc, test_auc))
