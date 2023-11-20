# The OGC method from the "From Cluster Assumption to Graph Convolution:
# Graph-based Semi-Supervised Learning Revisited" paper.
# ArXiv: https://arxiv.org/abs/2309.13599

# Datasets  CiteSeer  Cora   PubMed
# Acc       0.774     0.869  0.837
# Time      3.76      1.53   2.92

import argparse
import os.path as osp
import time
import warnings

import torch
import torch.nn.functional as F
from torch import Tensor

import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import one_hot

warnings.filterwarnings('ignore', '.*Sparse CSR tensor support.*')

decline = 0.9  # decline rate
eta_sup = 0.001  # learning rate for supervised loss
eta_W = 0.5  # learning rate for updating W
beta = 0.1  # moving probability that a node moves to neighbors
max_sim_tol = 0.995  # max label prediction similarity between iterations
max_patience = 2  # tolerance for consecutive similar test predictions

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')

transform = T.Compose([
    T.NormalizeFeatures(),
    T.GCNNorm(),
    T.ToSparseTensor(layout=torch.sparse_csr),
])
dataset = Planetoid(path, name=args.dataset, transform=transform)
data = dataset[0].to(device)

y_one_hot = one_hot(data.y, dataset.num_classes)
data.trainval_mask = data.train_mask | data.val_mask
# LIM track, else use trainval_mask to construct S
S = torch.diag(data.train_mask).float().to_sparse()
I_N = torch.eye(data.num_nodes).to_sparse(layout=torch.sparse_csr).to(device)

# Lazy random walk (also known as lazy graph convolution):
lazy_adj = beta * data.adj_t + (1 - beta) * I_N


class LinearNeuralNetwork(torch.nn.Module):
    def __init__(self, num_features: int, num_classes: int, bias: bool = True):
        super().__init__()
        self.W = torch.nn.Linear(num_features, num_classes, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.W(x)

    @torch.no_grad()
    def test(self, U: Tensor, y_one_hot: Tensor, data: Data):
        self.eval()
        out = self(U)

        loss = F.mse_loss(
            out[data.trainval_mask],
            y_one_hot[data.trainval_mask],
        )

        accs = []
        pred = out.argmax(dim=-1)
        for _, mask in data('trainval_mask', 'test_mask'):
            accs.append(float((pred[mask] == data.y[mask]).sum() / mask.sum()))

        return float(loss), accs[0], accs[1], pred

    def update_W(self, U: Tensor, y_one_hot: Tensor, data: Data):
        optimizer = torch.optim.SGD(self.parameters(), lr=eta_W)
        self.train()
        optimizer.zero_grad()
        pred = self(U)
        loss = F.mse_loss(pred[data.trainval_mask], y_one_hot[
            data.trainval_mask,
        ], reduction='sum')
        loss.backward()
        optimizer.step()
        return self(U).data, self.W.weight.data


model = LinearNeuralNetwork(
    num_features=dataset.num_features,
    num_classes=dataset.num_classes,
    bias=False,
).to(device)


def update_U(U: Tensor, y_one_hot: Tensor, pred: Tensor, W: Tensor):
    global eta_sup

    # Update the smoothness loss via LGC:
    U = lazy_adj @ U

    # Update the supervised loss via SEB:
    dU_sup = 2 * (S @ (-y_one_hot + pred)) @ W
    U = U - eta_sup * dU_sup

    eta_sup = eta_sup * decline
    return U


def ogc() -> float:
    U = data.x
    _, _, last_acc, last_pred = model.test(U, y_one_hot, data)

    patience = 0
    for i in range(1, 65):
        # Updating W by training a simple linear neural network:
        pred, W = model.update_W(U, y_one_hot, data)

        # Updating U by LGC and SEB jointly:
        U = update_U(U, y_one_hot, pred, W)

        loss, trainval_acc, test_acc, pred = model.test(U, y_one_hot, data)
        print(f'Epoch: {i:02d}, Loss: {loss:.4f}, '
              f'Train+Val Acc: {trainval_acc:.4f} Test Acc {test_acc:.4f}')

        sim_rate = float((pred == last_pred).sum()) / pred.size(0)
        if (sim_rate > max_sim_tol):
            patience += 1
            if (patience > max_patience):
                break

        last_acc, last_pred = test_acc, pred

    return last_acc


start_time = time.time()
test_acc = ogc()
print(f'Test Accuracy: {test_acc:.4f}')
print(f'Total Time: {time.time() - start_time:.4f}s')
