import argparse
import os.path as osp
import copy
from random import sample, shuffle

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
import networkx

from torch_geometric.utils import from_networkx, one_hot
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

parser = argparse.ArgumentParser()
# default parameter setting for CiteSeer.
parser.add_argument('--dataset', type=str, default='CiteSeer')
parser.add_argument('--degree', type=int, default=16)
parser.add_argument('--decline', type=float, default=1.0)
parser.add_argument('--decline_neg', type=float, default=1.0)
parser.add_argument('--alpha', type=float, default=0.15)
parser.add_argument('--lr', type=float, default=0.2)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--wd', type=float, default=1e-3)
parser.add_argument('--negative_rate', type=float, default=20.0)
parser.add_argument('--change_num', type=int, default=20)
args = parser.parse_args()
# for Cora and PubMed :
# python ggcm.py --dataset=Cora --decline_neg=0.5 --wd=1e-5
# python ggcm.py --dataset=PubMed --alpha=0.1 --decline_neg=0.5 --wd=2e-5 --change_num=80

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
transform = T.Compose([T.NormalizeFeatures(),
                       T.GCNNorm(),
                       T.ToSparseTensor(layout=torch.sparse_coo)])
dataset = Planetoid(path,
                    name=args.dataset,
                    transform=transform)
data = dataset[0].to(device)

num_nodes = data.x.size(0)
I_N = torch.eye(num_nodes).to_sparse().to(device)

edge_num = data.adj_t.indices().size(1)
avg_edge_num = int(args.negative_rate * (edge_num / num_nodes - 1))
# -1 for self loops
if avg_edge_num % 2 != 0:
    avg_edge_num += 1


def inver_graph_convolution(neg_graph: Tensor):
    change_num = args.change_num  # interchange edges for how many times
    if neg_graph is None:
        regular_graph = networkx.random_regular_graph(d=avg_edge_num, n=num_nodes)
        data_G = T.ToSparseTensor(layout=torch.sparse_coo)(from_networkx(regular_graph))
        neg_graph = data_G.adj_t.coalesce().to(device)
    else:
        # randomly modify neg_graph, perserving iters of nodes
        index_sample = sample(list(range(num_nodes * avg_edge_num)), change_num)
        neg_graph_ind = neg_graph.indices()
        ulis = neg_graph_ind[0][index_sample]
        vlis = neg_graph_ind[1][index_sample]

        ind = [[], []]
        val = []
        for i in range(change_num):
            u, v = ulis[i], vlis[i]
            ind[0].extend([u, v])
            ind[1].extend([v, u])
            val.extend([-1, -1])
        shuffle(vlis)
        for i in range(change_num):
            u, v = ulis[i], vlis[i]
            ind[0].extend([u, v])
            ind[1].extend([v, u])
            val.extend([1, 1])

        delta = torch.sparse_coo_tensor(ind,
                                        val,
                                        (num_nodes, num_nodes)).to(device)
        neg_graph += delta
        neg_graph = neg_graph.coalesce()

    # re-normalization trick
    return (2 * I_N - neg_graph) / (avg_edge_num + 2), neg_graph


def lazy_random_walk(adj: Tensor, beta: float) -> Tensor:
    return beta * adj + (1 - beta) * I_N


def ggcm(degree: int, alpha: float, decline: float, decline_neg: float) -> Tensor:
    beta = 1.0
    neg_beta = 1.0

    U = data.x
    adj = data.adj_t.to_sparse_coo()

    temp_sum = torch.zeros_like(U)
    neg_graph = None
    for _ in range(1, degree + 1):
        # lazy graph convolution (LGC)
        lazy_adj = lazy_random_walk(adj, beta)
        embedds = torch.sparse.mm(lazy_adj, U)

        # inverse graph convlution (IGC), lazy version
        neg_adj_nor, neg_graph = inver_graph_convolution(neg_graph)
        inv_lazy_adj = lazy_random_walk(neg_adj_nor, neg_beta)
        inv_embedds = torch.sparse.mm(inv_lazy_adj, U)

        # add for multi-scale version
        temp_sum += (embedds + inv_embedds) / 2.0
        U = embedds

        beta *= decline
        neg_beta *= decline_neg

    return alpha * data.x + (1 - alpha) * (temp_sum / degree)


class LinearNeuralNetwork(torch.nn.Module):
    def __init__(self, num_features: int, num_classes: int, bias: bool = True):
        super().__init__()
        self.W = torch.nn.Linear(num_features, num_classes, bias=bias)
        self.optimizer = optim.Adam(self.parameters(),
                                    lr=args.lr,
                                    weight_decay=args.wd)

    def forward(self, x: Tensor) -> Tensor:
        return self.W(x)

    @torch.no_grad()
    def test(self, X: Tensor, data: Data):
        self.eval()
        out = self(X)
        loss = F.cross_entropy(out[data.val_mask], data.y[data.val_mask])

        accs = []
        pred = out.argmax(dim=-1)
        for _, mask in data('val_mask', 'test_mask'):
            accs.append(float((pred[mask] == data.y[mask]).sum() / mask.sum()))

        return loss, accs[0], accs[1]

    def _train(self, embedds: Tensor, data: Data):
        self.train()
        self.optimizer.zero_grad()
        out = self(embedds)
        loss = F.cross_entropy(out[data.train_mask],
                               data.y[data.train_mask])
        loss.backward()
        self.optimizer.step()


model = LinearNeuralNetwork(num_features=data.x.size(1),
                            num_classes=one_hot(data.y).size(1),
                            bias=True).to(device)


def evaluate_embedds(X: Tensor, epochs: int) -> float:
    # Evaluate embedding by classification with the given split setting
    best_acc = -1
    best_model = None

    for i in range(1, epochs + 1):
        model._train(X, data)

        loss_val, acc_val, acc_test = model.test(X, data)
        if acc_val >= best_acc:
            best_acc, best_model = acc_val, copy.deepcopy(model)

        print(f'{i:03d} {loss_val:.4f} {acc_val:.3f} '
              f'acc_test = {acc_test:.3f}')

    loss_val, acc_val, acc_test = best_model.test(X, data)
    return acc_test


embedds = ggcm(args.degree, args.alpha, args.decline, args.decline_neg)
test_acc = evaluate_embedds(embedds, args.epochs)
print(f'Final Test Acc: {test_acc:.4f}')
