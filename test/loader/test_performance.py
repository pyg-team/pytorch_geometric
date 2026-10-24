import sys
import copy
import time
import tqdm
import random
# import shutil
import pytest
import os.path as osp

import torch
from torch.optim import Adam
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, GATConv
from torch_geometric.datasets import Planetoid, Reddit2
from torch_geometric.loader import NeighborSampler, NeighborLoader


class GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, 8, heads=8, dropout=0.6)
        self.conv2 = GATConv(8 * 8, out_channels, heads=1, dropout=0.6)

    def full(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def mini(self, x, adjs):
        x = F.dropout(x, p=0.6, training=self.training)
        edge_index, _, size = adjs[0]
        x_target = x[:size[1]]
        x = F.elu(self.conv1((x, x_target), edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        edge_index, _, size = adjs[1]
        x_target = x[:size[1]]
        x = self.conv2((x, x_target), edge_index)
        return x


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, 256)
        self.conv2 = SAGEConv(256, out_channels)

    def full(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def mini(self, x, adjs):
        edge_index, _, size = adjs[0]
        x_target = x[:size[1]]
        x = self.conv1((x, x_target), edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        edge_index, _, size = adjs[1]
        x_target = x[:size[1]]
        x = self.conv2((x, x_target), edge_index)
        return x


def train_full_batch(model, data, optimizer, device=None):
    data = copy.copy(data).to(device)
    train_acc = val_acc = test_acc = 0.
    for _ in tqdm.tqdm(range(200)):
        model.train()
        optimizer.zero_grad()
        out = model.full(data.x, data.edge_index)[data.train_mask]
        loss = F.cross_entropy(out, data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model.full(data.x, data.edge_index)
            cor = out.argmax(dim=-1) == data.y

            mask = data.train_mask
            train_acc = max(int(cor[mask].sum()) / int(mask.sum()), train_acc)

            mask = data.val_mask
            val_acc = max(int(cor[mask].sum()) / int(mask.sum()), val_acc)

            mask = data.test_mask
            test_acc = max(int(cor[mask].sum()) / int(mask.sum()), test_acc)

    return train_acc, val_acc, test_acc


def train_neighbor_sampler(model, data, train_loader, val_loader, test_loader,
                           optimizer, device=None):
    data = copy.copy(data).to(device)
    train_acc = val_acc = test_acc = 0.
    for _ in tqdm.tqdm(range(200)):
        model.train()

        for batch_size, n_id, adjs in train_loader:
            adjs = [adj.to(device) for adj in adjs]
            optimizer.zero_grad()
            out = model.mini(data.x[n_id], adjs)
            loss = F.cross_entropy(out, data.y[n_id[:batch_size]])
            print(loss)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            train_correct = train_examples = 0.
            for batch_size, n_id, adjs in train_loader:
                adjs = [adj.to(device) for adj in adjs]
                out = model.mini(data.x[n_id], adjs)
                correct = out.argmax(dim=-1) == data.y[n_id[:batch_size]]
                train_correct += int(correct.sum())
                train_examples += batch_size
                print(train_correct / train_examples)
            train_acc = max(train_correct / train_examples, train_acc)

            val_correct = val_examples = 0.
            for batch_size, n_id, adjs in val_loader:
                adjs = [adj.to(device) for adj in adjs]
                out = model.mini(data.x[n_id], adjs)
                correct = out.argmax(dim=-1) == data.y[n_id[:batch_size]]
                val_correct += int(correct.sum())
                val_examples += batch_size
            val_acc = max(val_correct / val_examples, val_acc)

            test_correct = test_examples = 0.
            for batch_size, n_id, adjs in test_loader:
                adjs = [adj.to(device) for adj in adjs]
                out = model.mini(data.x[n_id], adjs)
                correct = out.argmax(dim=-1) == data.y[n_id[:batch_size]]
                test_correct += int(correct.sum())
                test_examples += batch_size
            test_acc = max(test_correct / test_examples, test_acc)

    return train_acc, val_acc, test_acc


def train_neighbor_loader(model, train_loader, val_loader, test_loader,
                          optimizer, device=None):
    train_acc = val_acc = test_acc = 0.
    for _ in tqdm.tqdm(range(200)):
        model.train()
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model.full(data.x, data.edge_index)[:data.batch_size]
            loss = F.cross_entropy(out, data.y[:data.batch_size])
            print(loss)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            train_correct = train_examples = 0.
            for data in train_loader:
                data = data.to(device)
                out = model.full(data.x, data.edge_index)[:data.batch_size]
                correct = out.argmax(dim=-1) == data.y[:data.batch_size]
                train_correct += int(correct.sum())
                train_examples += data.batch_size
                print(train_correct / train_examples)
            train_acc = max(train_correct / train_examples, train_acc)

            val_correct = val_examples = 0.
            for data in val_loader:
                data = data.to(device)
                out = model.full(data.x, data.edge_index)[:data.batch_size]
                correct = out.argmax(dim=-1) == data.y[:data.batch_size]
                val_correct += int(correct.sum())
                val_examples += data.batch_size
            val_acc = max(val_correct / val_examples, val_acc)

            test_correct = test_examples = 0.
            for data in test_loader:
                data = data.to(device)
                out = model.full(data.x, data.edge_index)[:data.batch_size]
                correct = out.argmax(dim=-1) == data.y[:data.batch_size]
                test_correct += int(correct.sum())
                test_examples += data.batch_size
            test_acc = max(test_correct / test_examples, test_acc)

    return train_acc, val_acc, test_acc


def test_performance_of_loaders_on_gat_and_cora():
    return
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    root_dir = osp.join('/', 'tmp', str(random.randrange(sys.maxsize)))
    root_dir = '/tmp/bladawdhihd'
    dataset = Planetoid(root_dir, name='Cora', transform=T.NormalizeFeatures())
    # shutil.rmtree(root_dir)
    data = dataset[0]

    torch.manual_seed(1234)
    model = GAT(dataset.num_features, dataset.num_classes).to(device)
    optimizer = Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    t = time.perf_counter()
    out = train_full_batch(model, data, optimizer, device)
    print(time.perf_counter() - t)
    print(out)

    ###########################################################################

    train_loader = NeighborLoader(data, num_neighbors=[5, 5], shuffle=True,
                                  input_nodes=data.train_mask, batch_size=128,
                                  num_workers=6, persistent_workers=True)
    val_loader = NeighborLoader(data, num_neighbors=[5, 5], shuffle=True,
                                input_nodes=data.val_mask, batch_size=128,
                                num_workers=6, persistent_workers=True)
    test_loader = NeighborLoader(data, num_neighbors=[5, 5], shuffle=True,
                                 input_nodes=data.val_mask, batch_size=128,
                                 num_workers=6, persistent_workers=True)

    torch.manual_seed(1234)
    model = GAT(dataset.num_features, dataset.num_classes).to(device)
    optimizer = Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    t = time.perf_counter()
    out = train_neighbor_loader(model, train_loader, val_loader, test_loader,
                                optimizer, device)
    print(time.perf_counter() - t)
    print(out)

    ###########################################################################

    train_loader = NeighborSampler(data.edge_index, sizes=[5, 5], shuffle=True,
                                   node_idx=data.train_mask, batch_size=128,
                                   num_workers=6, persistent_workers=True)
    val_loader = NeighborSampler(data.edge_index, sizes=[5, 5], shuffle=True,
                                 node_idx=data.val_mask, batch_size=128,
                                 num_workers=6, persistent_workers=True)
    test_loader = NeighborSampler(data.edge_index, sizes=[5, 5], shuffle=True,
                                  node_idx=data.test_mask, batch_size=128,
                                  num_workers=6, persistent_workers=True)

    torch.manual_seed(1234)
    model = GAT(dataset.num_features, dataset.num_classes).to(device)
    optimizer = Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    t = time.perf_counter()
    out = train_neighbor_sampler(model, data, train_loader, val_loader,
                                 test_loader, optimizer, device)
    print(time.perf_counter() - t)
    print(out)


def test_performance_of_loaders_on_graphsage_and_reddit():
    return
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    root_dir = osp.join('/', 'tmp', str(random.randrange(sys.maxsize)))
    root_dir = '/data/datasets/Reddit2'
    dataset = Reddit2(root_dir)
    # shutil.rmtree(root_dir)
    data = dataset[0]

    print(dataset)
    print(data)

    # Evaluate `NeighborSampler` ##############################################

    # kwargs = {
    #     'edge_index': data.edge_index,
    #     'sizes': [25, 10],
    #     'batch_size': 1024,
    #     'shuffle': True,
    #     'num_workers': 6,
    #     'persistent_workers': True,
    # }

    # train_loader = NeighborSampler(node_idx=data.train_mask, **kwargs)
    # val_loader = NeighborSampler(node_idx=data.val_mask, **kwargs)
    # test_loader = NeighborSampler(node_idx=data.test_mask, **kwargs)

    # torch.manual_seed(1234)
    # model = GraphSAGE(dataset.num_features, dataset.num_classes).to(device)
    # optimizer = Adam(model.parameters(), lr=0.01)
    # t = time.perf_counter()
    # out = train_neighbor_sampler(model, data, train_loader, val_loader,
    #                              test_loader, optimizer, device)
    # print(time.perf_counter() - t)
    # print(out)
    # return

    # Evaluate `NeighborLoader` ###############################################

    kwargs = {
        'data': data,
        'num_neighbors': [25, 10],
        'batch_size': 1024,
        'shuffle': True,
        'num_workers': 6,
        'persistent_workers': True,
    }

    train_loader = NeighborLoader(input_nodes=data.train_mask, **kwargs)
    val_loader = NeighborLoader(input_nodes=data.val_mask, **kwargs)
    test_loader = NeighborLoader(input_nodes=data.test_mask, **kwargs)

    torch.manual_seed(1234)
    model = GAT(dataset.num_features, dataset.num_classes).to(device)
    optimizer = Adam(model.parameters(), lr=0.01)
    t = time.perf_counter()
    out = train_neighbor_loader(model, train_loader, val_loader, test_loader,
                                optimizer, device)
    print(time.perf_counter() - t)
    print(out)
