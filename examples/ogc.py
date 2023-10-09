import argparse
import os.path as osp
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

decline = 0.9        # decline rate
eta_sup = 0.001      # learning rate for supervised loss
eta_W = 0.5          # learning rate for updating W
beta = 0.1           # moving probability that a node moves to neighbors
max_sim_tol = 0.995  # max label prediction similarity between iterations
max_patience = 2     # tolerance for consecutive similar test predictions

# Datasets          CiteSeer            Cora                PubMed
# Acc               0.774               0.869               0.837
# Time              3.76                1.53                2.92

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',
                    type=str,
                    default='Cora',
                    choices=['Cora', 'CiteSeer', 'PubMed'],
                    help='Dataset to use.')
args, _ = parser.parse_known_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')

transform = T.Compose([T.NormalizeFeatures(), T.GCNNorm(), T.ToSparseTensor()])
dataset = Planetoid(path,
                    name=args.dataset,
                    transform=transform)
data = dataset[0].to(device)

y_one_hot = F.one_hot(data.y).float()
data.tv_mask = data.train_mask + data.val_mask
S = torch.diag(data.train_mask).float().to_sparse()
# LIM track, else use tv_mask to construct S
I_N = torch.eye(data.x.shape[0]).to_sparse().to(device)


class LinearNeuralNetwork(nn.Module):
    def __init__(self, nfeat, nclass, bias=True):
        super(LinearNeuralNetwork, self).__init__()
        self.W = nn.Linear(nfeat, nclass, bias=bias)

    def forward(self, x):
        return self.W(x)

    @torch.no_grad()
    def test(self, U, y_one_hot, data):
        self.eval()
        output = self(U)
        pred = output.argmax(dim=-1)
        loss_tv = float(F.mse_loss(output[data.tv_mask],
                                   y_one_hot[data.tv_mask]))
        accs = []
        for _, mask in data('tv_mask', 'test_mask'):
            accs.append(float((pred[mask] == data.y[mask]).sum() / mask.sum()))
        return loss_tv, accs[0], accs[1], pred

    def update_W(self, U, y_one_hot, data):
        optimizer = optim.SGD(self.parameters(), lr=eta_W)
        self.train()
        optimizer.zero_grad()
        pred = self(U)
        loss_tv = F.mse_loss(pred[data.tv_mask],
                             y_one_hot[data.tv_mask],
                             reduction='sum')
        loss_tv.backward()
        optimizer.step()
        return self(U).data, self.W.weight.data


linear_clf = LinearNeuralNetwork(nfeat=data.x.shape[1],
                                 nclass=y_one_hot.shape[1],
                                 bias=False).to(device)
# lazy random walk (also known as lazy graph convolution)
lazy_adj = beta * data.adj_t.to_sparse_coo() + (1 - beta) * I_N


def update_U(U, Y, predY, W):
    global eta_sup

    # ------ update the smoothness loss via LGC ------
    U = torch.sparse.mm(lazy_adj, U)

    # ------ update the supervised loss via SEB ------
    dU_sup = 2 * torch.mm(torch.sparse.mm(S, -Y + predY), W)
    U = U - eta_sup * dU_sup

    eta_sup = eta_sup * decline
    return U


def OGC():
    patience = 0
    U = data.x
    test_res = linear_clf.test(U, y_one_hot, data)
    _, _, last_acc, last_outp = test_res
    for i in range(64):
        # updating W by training a simple linear neural network
        predY, W = linear_clf.update_W(U, y_one_hot, data)

        # updating U by LGC and SEB jointly
        U = update_U(U, y_one_hot, predY, W)

        test_res = linear_clf.test(U, y_one_hot, data)
        loss_tv, acc_tv, acc_test, pred = test_res
        print(f'epoch {i} loss_train_val {loss_tv:.4f} ' +
              f'acc_train_val {acc_tv:.4f} acc_test {acc_test:.4f}')

        sim_rate = float((pred == last_outp).sum()) / pred.shape[0]
        if (sim_rate > max_sim_tol):
            patience += 1
            if (patience > max_patience):
                break

        last_acc = acc_test
        last_outp = pred
    return last_acc


start_time = time.time()
res = OGC()
time_tot = time.time() - start_time

print(f'Test Acc: {res:.4f}')
print(f'Total Time: {time_tot:.4f}')
