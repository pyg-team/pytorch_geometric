import argparse
import time
from math import ceil

import matplotlib.pyplot as plt
import torch
from ogb.nodeproppred import PygNodePropPredDataset

from torch_geometric.data import Data
from torch_geometric.loader import IBMBNodeLoader, NeighborLoader
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_remaining_self_loops, to_undirected


class GraphPreprocess:
    def __init__(self, self_loop: bool = True,
                 transform_to_undirected: bool = True):
        self.self_loop = self_loop
        self.to_undirected = transform_to_undirected

    def __call__(self, graph: Data):
        graph.y = graph.y.reshape(-1)
        graph.y = torch.nan_to_num(graph.y, nan=-1)
        graph.y = graph.y.to(torch.long)

        if self.self_loop:
            edge_index, _ = add_remaining_self_loops(graph.edge_index,
                                                     num_nodes=graph.num_nodes)
        else:
            edge_index = graph.edge_index

        if self.to_undirected:
            edge_index = to_undirected(edge_index, num_nodes=graph.num_nodes)

        graph.edge_index = edge_index
        return graph


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_channels,
                 num_layers, normalize, add_self_loops):
        super(GCN, self).__init__()
        self.conv_layers = torch.nn.ModuleList([])
        self.bns = torch.nn.ModuleList([])

        self.num_layers = num_layers
        for i in range(num_layers):
            in_channels = num_node_features if i == 0 else hidden_channels
            self.conv_layers.append(
                GCNConv(in_channels=in_channels, out_channels=hidden_channels,
                        normalize=normalize, add_self_loops=add_self_loops))
            self.bns.append(
                torch.nn.LayerNorm(hidden_channels, elementwise_affine=True))

        self.lastlayer = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for i, l in enumerate(self.conv_layers):
            x = l(x, edge_index)
            x = self.bns[i](x)
            x = torch.nn.functional.dropout(torch.relu(x), p=0.5,
                                            training=self.training)
        x = self.lastlayer(x)

        return x

    def reset_parameters(self):
        for c in self.conv_layers:
            c.reset_parameters()
        for c in self.bns:
            c.reset_parameters()
        self.lastlayer.reset_parameters()


def train(model, train_loader, optimizer):
    model.train()

    corrects = 0
    total_loss = 0
    num_nodes = 0
    for data in train_loader:
        data = data.to(device)
        y = data.y[data.output_node_mask]

        optimizer.zero_grad()
        outputs = model(data)[data.output_node_mask]
        loss = torch.nn.CrossEntropyLoss()(outputs, y)
        loss.backward()

        optimizer.step()
        pred = torch.argmax(outputs.detach(), dim=1)
        corrects += pred.eq(y).sum()
        total_loss += loss * len(y)
        num_nodes += len(y)
    return total_loss / num_nodes, corrects / num_nodes


@torch.no_grad()
def test(model, loader, scheduler):
    model.eval()

    corrects = 0
    total_loss = 0
    num_nodes = 0
    for data in loader:
        data = data.to(device)
        y = data.y[data.output_node_mask]

        outputs = model(data)[data.output_node_mask]
        loss = torch.nn.CrossEntropyLoss()(outputs, y)
        if scheduler is not None:
            scheduler.step(loss)
        pred = torch.argmax(outputs.detach(), dim=1)
        corrects += pred.eq(y).sum().item()
        total_loss += loss.item() * len(y)
        num_nodes += len(y)
    return total_loss / num_nodes, corrects / num_nodes


def train_neighborloader(model, train_loader, optimizer):
    model.train()

    corrects = 0
    total_loss = 0
    num_nodes = 0
    for data in train_loader:
        data = data.to(device)
        batch_size = data.batch_size
        y = data.y[:batch_size]

        optimizer.zero_grad()
        outputs = model(data)[:batch_size]
        loss = torch.nn.CrossEntropyLoss()(outputs, y)
        loss.backward()

        optimizer.step()
        pred = torch.argmax(outputs.detach(), dim=1)
        corrects += pred.eq(y).sum()
        total_loss += loss * len(y)
        num_nodes += len(y)
    return total_loss / num_nodes, corrects / num_nodes


@torch.no_grad()
def test_neighborloader(model, loader, scheduler):
    model.eval()

    corrects = 0
    total_loss = 0
    num_nodes = 0
    for data in loader:
        data = data.to(device)
        batch_size = data.batch_size
        y = data.y[:batch_size]

        outputs = model(data)[:batch_size]

        loss = torch.nn.CrossEntropyLoss()(outputs, y)
        if scheduler is not None:
            scheduler.step(loss)
        pred = torch.argmax(outputs.detach(), dim=1)
        corrects += pred.eq(y).sum().item()
        total_loss += loss.item() * len(y)
        num_nodes += len(y)
    return total_loss / num_nodes, corrects / num_nodes


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_aux', type=int, default=16,
                        help='number of auxiliary nodes per output node')
    parser.add_argument('--num_train_prime', type=int, default=9000,
                        help='number of output nodes in a training batch')
    parser.add_argument(
        '--num_val_prime', type=int, default=18000,
        help='number of output nodes in a validation batch,'
        'typically larger than num_train_prime because '
        'it requires no backward propagation')
    parser.add_argument('--alpha', type=float, default=0.25,
                        help='parameter controlling PPR calculation')
    parser.add_argument('--eps', type=float, default=2.e-4,
                        help='parameter controlling PPR calculation')
    parser.add_argument('--hidden_channels', type=int, default=256,
                        help='hidden dimension for the GNN')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='number of layers for the GNN')
    parser.add_argument('--weight_decay', type=float, default=1.e-4)
    return parser.parse_args()


if __name__ == '__main__':
    args = args_parser()

    dataset = PygNodePropPredDataset(name="ogbn-arxiv", root='./datasets',
                                     pre_transform=GraphPreprocess())
    split_idx = dataset.get_idx_split()
    graph = dataset[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ibmb_timing = []
    ibmb_acc = []
    baseline_timing = []
    baseline_acc = []

    model = GCN(graph.num_node_features,
                graph.y.max().item() + 1, args.hidden_channels,
                args.num_layers, normalize=False,
                add_self_loops=False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.e-3,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.33, patience=50, min_lr=1.e-5)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ibmb_start_time = time.time()
    # IBMB
    train_loader = IBMBNodeLoader(
        graph, batch_order='order', input_nodes=split_idx["train"],
        return_edge_index_type='adj', num_auxiliary_nodes=args.num_aux,
        num_nodes_per_batch=args.num_train_prime, alpha=args.alpha,
        eps=args.eps, batch_size=1, shuffle=False)
    val_loader = IBMBNodeLoader(
        graph, batch_order='order', input_nodes=split_idx["valid"],
        return_edge_index_type='adj', num_auxiliary_nodes=args.num_aux,
        num_nodes_per_batch=args.num_val_prime, alpha=args.alpha, eps=args.eps,
        batch_size=1, shuffle=False)
    test_loader = IBMBNodeLoader(
        graph, batch_order='order', input_nodes=split_idx["test"],
        return_edge_index_type='adj', num_auxiliary_nodes=args.num_aux,
        num_nodes_per_batch=args.num_val_prime, alpha=args.alpha, eps=args.eps,
        batch_size=1, shuffle=False)

    print("=========== Start IBMB training ===========")
    for epoch in range(300):
        train_loss, train_acc = train(model, train_loader, optimizer)
        val_loss, val_acc = test(model, val_loader, scheduler)
        test_loss, test_acc = test(model, test_loader, None)
        print(f'Epoch: {epoch}, '
              f'train loss: {train_loss:.4f}, train acc: {train_acc:.4f}, '
              f'val loss: {val_loss:.4f}, val acc: {val_acc:.4f}, '
              f'test loss: {test_loss:.4f}, test acc: {test_acc:.4f}')

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ibmb_timing.append(time.time() - ibmb_start_time)
        ibmb_acc.append(val_acc)

    model = GCN(graph.num_node_features,
                graph.y.max().item() + 1, args.hidden_channels,
                args.num_layers, normalize=True,
                add_self_loops=False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.e-3,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.33, patience=50, min_lr=1.e-5)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    baseline_start_time = time.time()
    # some hyperparameters we used,
    # in principle it should consume similar GPU as the IBMB loader
    num_nodes = [3, 3, 3]
    # Neighbor Loader
    # the number of batches should be the same as IBMB loader
    train_loader = NeighborLoader(
        graph, num_neighbors=num_nodes, input_nodes=split_idx["train"],
        batch_size=ceil(len(split_idx["train"]) / len(train_loader)),
        shuffle=True)
    val_loader = NeighborLoader(
        graph, num_neighbors=num_nodes, input_nodes=split_idx["valid"],
        batch_size=ceil(len(split_idx["train"]) / len(val_loader)),
        shuffle=True)
    test_loader = NeighborLoader(
        graph, num_neighbors=num_nodes, input_nodes=split_idx["test"],
        batch_size=ceil(len(split_idx["test"]) / len(test_loader)),
        shuffle=True)

    print("=========== Start baseline training ===========")
    for epoch in range(300):
        train_loss, train_acc = train_neighborloader(model, train_loader,
                                                     optimizer)
        val_loss, val_acc = test_neighborloader(model, val_loader, scheduler)
        best_loss, best_acc = test_neighborloader(model, val_loader, None)
        print(f'Epoch: {epoch}, '
              f'train loss: {train_loss:.4f}, train acc: {train_acc:.4f}, '
              f'val loss: {val_loss:.4f}, val acc: {val_acc:.4f}, '
              f'test loss: {best_loss:.4f}, test acc: {best_acc:.4f}')
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        baseline_timing.append(time.time() - baseline_start_time)
        baseline_acc.append(val_acc)

    fig = plt.figure()
    plt.plot(ibmb_timing, ibmb_acc, label="IBMB")
    plt.plot(baseline_timing, baseline_acc, label="Baseline")
    plt.legend()
    plt.xscale("log")
    # plt.ylim(66, 72)
    fig.savefig('curves.png', dpi=400)
    plt.close()
