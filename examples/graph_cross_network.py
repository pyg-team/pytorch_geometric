import argparse
import sys

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import random_split

from torch_geometric.data import Dataset
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GraphCrossNet
from torch_geometric.utils import add_remaining_self_loops, degree


class process_dataset(Dataset):
    def __init__(self, root, name):
        super().__init__(root)

        dataset = TUDataset(root, name)
        self.graphs = [x for x in dataset]
        self.num_classes = dataset.num_classes

        self.augment_self_loops()
        self.degree_as_feature()

    def degree_as_feature(self):
        min_degree, max_degree = self.get_min_max_degrees()
        vec_len = max_degree - min_degree + 1
        for g in self.graphs:
            num_nodes = g.num_nodes
            node_feat = torch.zeros((num_nodes, vec_len))
            if g.is_undirected():
                d = degree(g.edge_index.flatten(), num_nodes, dtype=torch.long)
            else:
                d = degree(g.edge_index[1, :], num_nodes, dtype=torch.long)
            node_feat[torch.arange(num_nodes), d - min_degree] = 1.
            if g.x is not None:
                g.x = torch.cat((g.x, node_feat), dim=1)
            else:
                g.x = node_feat

    def augment_self_loops(self):
        for i in range(len(self.graphs)):
            edge_index = add_remaining_self_loops(self.graphs[i].edge_index)
            self.graphs[i].edge_index = edge_index[0]

    def get_min_max_degrees(self):
        min_degree = sys.maxsize
        max_degree = 0
        for g in self.graphs:
            if g.is_undirected():
                degrees = degree(g.edge_index.flatten()).long()
            else:
                degrees = degree(g.edge_index[1, :]).long()
            min_degree = min(min_degree, degrees.min().item())
            max_degree = max(max_degree, degrees.max().item())
        return min_degree, max_degree

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]

    def statistics(self):
        return self.graphs[0].x.shape[1], self.num_classes


class GraphClassifier(torch.nn.Module):
    """
    Description
    -----------
    Graph Classifier for graph classification.
    GXN + MLP
    """
    def __init__(self, args):
        super().__init__()
        self.gxn = GraphCrossNet(
            in_dim=args.in_dim, out_dim=args.embed_dim,
            edge_feat_dim=args.edge_feat_dim, hidden_dim=args.hidden_dim,
            pool_ratios=args.pool_ratios, readout_nodes=args.readout_nodes,
            conv1d_dims=args.conv1d_dims, conv1d_kws=args.conv1d_kws,
            cross_weight=args.cross_weight, fuse_weight=args.fuse_weight)
        self.lin1 = torch.nn.Linear(args.embed_dim, args.final_hidden_dim)
        self.lin2 = torch.nn.Linear(args.final_hidden_dim, args.out_dim)
        self.dropout = args.dropout

    def forward(self, graph):
        embed, logits1, logits2 = self.gxn(graph)
        logits = F.relu(self.lin1(embed))
        if self.dropout > 0:
            logits = F.dropout(logits, p=self.dropout, training=self.training)
        logits = self.lin2(logits)
        return F.log_softmax(logits, dim=1), logits1, logits2


def compute_loss(cls_logits: Tensor, labels: Tensor, logits_s1: Tensor,
                 logits_s2: Tensor, epoch: int, total_epochs: int,
                 device: torch.device):
    # classification loss
    classify_loss = F.nll_loss(cls_logits, labels.to(device))

    # loss for vertex infomax pooling
    scale1, scale2 = logits_s1.size(0) // 2, logits_s2.size(0) // 2
    s1_label_t, s1_label_f = torch.ones(scale1), torch.zeros(scale1)
    s2_label_t, s2_label_f = torch.ones(scale2), torch.zeros(scale2)
    s1_label = torch.cat((s1_label_t, s1_label_f), dim=0).to(device)
    s2_label = torch.cat((s2_label_t, s2_label_f), dim=0).to(device)

    pool_loss_s1 = F.binary_cross_entropy_with_logits(logits_s1, s1_label)
    pool_loss_s2 = F.binary_cross_entropy_with_logits(logits_s2, s2_label)
    pool_loss = (pool_loss_s1 + pool_loss_s2) / 2

    loss = classify_loss + (2 - epoch / total_epochs) * pool_loss

    return loss


def train(model: torch.nn.Module, optimizer, trainloader, device, curr_epoch,
          total_epochs):
    model.train()

    total_loss = 0.
    num_batches = len(trainloader)

    for batch in trainloader:
        optimizer.zero_grad()
        batch.to(device)
        out, l1, l2 = model(batch)
        loss = compute_loss(out, batch.y.long(), l1, l2, curr_epoch,
                            total_epochs, device)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / num_batches


@torch.no_grad()
def test(model: torch.nn.Module, loader, device):
    model.eval()

    correct = 0.
    num_graphs = 0

    for batch in loader:
        batch.to(device)
        num_graphs += batch.num_graphs
        out, _, _ = model(batch)
        pred = out.argmax(dim=1)
        correct += pred.eq(batch.y.long()).sum().item()

    return correct / num_graphs


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Graph Cross Network")
    parser.add_argument("--pool_ratios", nargs="+", type=float,
                        help="The pooling ratios used in graph cross layers")
    parser.add_argument("--hidden_dim", type=int, default=96,
                        help="The number of hidden channels in GXN")
    parser.add_argument("--cross_weight", type=float, default=1.0,
                        help="Weight parameter used in graph cross layer")
    parser.add_argument("--fuse_weight", type=float, default=0.9,
                        help="Weight parameter for feature fusion")
    parser.add_argument("--num_cross_layers", type=int, default=2,
                        help="The number of graph corss layers")
    parser.add_argument(
        "--readout_nodes", type=int, default=30,
        help="Number of nodes for each graph after \
                            final graph pooling")
    parser.add_argument(
        "--conv1d_dims", nargs="+", type=int,
        help="Number of channels in conv operations \
                            in the end of graph cross net")
    parser.add_argument("--conv1d_kws", nargs="+", type=int,
                        help="Kernel sizes of conv1d operations")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout rate")
    parser.add_argument("--embed_dim", type=int, default=1024,
                        help="Number of channels of graph embedding")
    parser.add_argument(
        "--final_hidden_dim", type=int, default=128,
        help="The number of hidden channels in final \
                            dense layers")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-3,
                        help="Weight decay rate")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of training epochs")
    parser.add_argument("--dataset", type=str, default="DD",
                        help="TUDataset used for training")

    args = parser.parse_args()

    # Default value for list hyper-parameters
    if not args.pool_ratios or len(args.pool_ratios) < 2:
        args.pool_ratios = [0.8, 0.7]
    if not args.conv1d_dims or len(args.conv1d_dims) < 2:
        args.conv1d_dims = [16, 32]
    if not args.conv1d_kws or len(args.conv1d_kws) < 1:
        args.conv1d_kws = [5]

    # Prepare graph data
    dataset = process_dataset("./datasets", args.dataset)

    # Retrieve train/validation/test index
    num_training = int(len(dataset) * 0.9)
    num_test = len(dataset) - num_training
    train_set, test_set = random_split(dataset, [num_training, num_test])
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              num_workers=1, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             num_workers=1)

    # Create model
    args.in_dim, args.out_dim = dataset.statistics()
    args.edge_feat_dim = 0  # No edge feature in datasets that we use.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphClassifier(args).to(device)

    # Create training components
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    # training epochs
    best_test_acc, best_epoch = 0.0, -1
    for e in range(args.epochs):
        train_loss = train(model, optimizer, train_loader, device, e,
                           args.epochs)
        test_acc = test(model, test_loader, device)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = e + 1

        log_format = "Epoch {}: loss={:.4f}, test_acc={:.4f}"
        print(log_format.format(e + 1, train_loss, test_acc))

    log_format = "Best Epoch {}, final test acc {:.4f}"
    print(log_format.format(best_epoch, best_test_acc))
