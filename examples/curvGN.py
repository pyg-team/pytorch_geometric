import os.path as osp
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import ConvCurv, compute_ricci_curvature
from torch_geometric.utils import add_self_loops, remove_self_loops
import torch_geometric.transforms as T


d_name = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', d_name)
dataset = Planetoid(path, d_name, transform=T.NormalizeFeatures())
data = dataset[0]
ricci_cur = compute_ricci_curvature(dataset, d_name)
w_mul = [i[2] for i in ricci_cur]
w_mul = w_mul + [0 for i in range(data.x.size(0))]
w_mul = torch.tensor(w_mul, dtype=torch.float)
data.edge_index, _ = remove_self_loops(data.edge_index)
data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.x.size(0))
data.w_mul = w_mul.view(-1, 1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data.w_mul = data.w_mul.to(device)


class Net(torch.nn.Module):
    def __init__(self, w_mul):
        super(Net, self).__init__()
        self.convcurv = ConvCurv(dataset.num_features,
                                 dataset.num_classes, w_mul)

    def forward(self, data):
        x = self.convcurv(data)
        return x


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, data = Net(data.w_mul).to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0005)


def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model(data)[data.train_mask],
               data.y[data.train_mask]).backward()
    optimizer.step()


def test():
    model.eval()
    logits, accs = model(data), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


best_val_acc = test_acc = 0
for epoch in range(1, 201):
    train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, train_acc, best_val_acc, test_acc))
