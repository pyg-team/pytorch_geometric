import argparse

import torch
from ogb.nodeproppred import PygNodePropPredDataset
from torch_sparse import matmul

from torch_geometric.data import Data
from torch_geometric.loader import IBMBNodeLoader
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


class MyGCNConv(GCNConv):
    def forward(self, x, edge_index):
        x = self.lin(x)
        out = matmul(edge_index, x, reduce=self.aggr)
        return out


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_channels,
                 num_layers):
        super(GCN, self).__init__()
        self.conv_layers = torch.nn.ModuleList([])
        self.bns = torch.nn.ModuleList([])

        self.num_layers = num_layers
        for i in range(num_layers):
            in_channels = num_node_features if i == 0 else hidden_channels
            self.conv_layers.append(
                MyGCNConv(in_channels=in_channels,
                          out_channels=hidden_channels))
            self.bns.append(
                torch.nn.LayerNorm(hidden_channels, elementwise_affine=True))

        self.lastlayer = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        x, adjs, prime_index = data.x, data.edge_index, data.output_node_mask

        for i, l in enumerate(self.conv_layers):
            if i == self.num_layers - 1 and prime_index is not None:
                x = l(x, adjs[prime_index, :])
            else:
                x = l(x, adjs)
            x = self.bns[i](x)
            x = torch.nn.functional.dropout(torch.relu(x), p=0.5,
                                            training=self.training)
        x = self.lastlayer(x)

        return x.log_softmax(dim=-1)


def train(model, train_loader, optimizer):
    model.train()

    corrects = 0
    total_loss = 0
    num_nodes = 0
    for data in train_loader:
        data = data.to(device)
        y = data.y[data.output_node_mask]

        optimizer.zero_grad()
        outputs = model(data)
        loss = torch.nn.functional.nll_loss(outputs, y)
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

        outputs = model(data)
        loss = torch.nn.functional.nll_loss(outputs, y)
        if scheduler is not None:
            scheduler.step(loss)
        pred = torch.argmax(outputs.detach(), dim=1)
        corrects += pred.eq(y).sum()
        total_loss += loss.item() * len(y)
        num_nodes += len(y)
    return total_loss / num_nodes, corrects / num_nodes


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_aux', type=int, default=16,
                        help='number of auxiliary nodes per output node')
    parser.add_argument('--num_train_prime', type=int, default=9000,
                        help='number of output nodes in a training batch')
    parser.add_argument('--num_val_prime', type=int, default=18000,
                        help='number of output nodes in a validation batch')
    parser.add_argument('--alpha', type=float, default=0.2,
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

    train_loader = IBMBNodeLoader(
        graph, batch_order='order', output_indices=split_idx["train"],
        return_edge_index_type='adj',
        num_auxiliary_node_per_output=args.num_aux,
        num_output_nodes_per_batch=args.num_train_prime, alpha=args.alpha,
        batch_size=1, shuffle=False)
    val_loader = IBMBNodeLoader(graph, batch_order='order',
                                output_indices=split_idx["valid"],
                                return_edge_index_type='adj',
                                num_auxiliary_node_per_output=args.num_aux,
                                num_output_nodes_per_batch=args.num_val_prime,
                                alpha=args.alpha, batch_size=1, shuffle=False)
    test_loader = IBMBNodeLoader(graph, batch_order='order',
                                 output_indices=split_idx["test"],
                                 return_edge_index_type='adj',
                                 num_auxiliary_node_per_output=args.num_aux,
                                 num_output_nodes_per_batch=args.num_val_prime,
                                 alpha=args.alpha, batch_size=1, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(graph.num_node_features,
                graph.y.max().item() + 1, args.hidden_channels,
                args.num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.e-3,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.33, patience=30, cooldown=10,
        min_lr=1.e-4)

    for epoch in range(300):
        train_loss, train_acc = train(model, train_loader, optimizer)
        val_loss, val_acc = test(model, val_loader, scheduler)
        best_loss, best_acc = test(model, val_loader, None)
        print(f'Epoch: {epoch}, '
              f'train loss: {train_loss:.4f}, train acc: {train_acc:.4f}, '
              f'val loss: {val_loss:.4f}, val acc: {val_acc:.4f}, '
              f'test loss: {best_loss:.4f}, test acc: {best_acc:.4f}')
