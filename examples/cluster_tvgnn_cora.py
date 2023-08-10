import os.path as osp

import torch
from sklearn.metrics.cluster import normalized_mutual_info_score as NMI
from torch import Tensor

import torch_geometric.transforms as transforms
from torch_geometric import utils
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GTVConv, dense_asymcheegercut_pool
from torch_geometric.nn.models.mlp import MLP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Cora')
dataset = Planetoid(path, 'Cora', transform=transforms.NormalizeFeatures())
data = dataset[0]
data = data.to(device)


class Net(torch.nn.Module):
    def __init__(self, in_channels, n_clusters, mp_layers=2, mp_channels=512,
                 mlp_hidden_layers=1, mlp_hidden_channels=256,
                 delta_coeff=0.3, eps=1e-3, totvar_coeff=0.8,
                 balance_coeff=0.5):
        super().__init__()

        self.totvar_coeff = totvar_coeff
        self.balance_coeff = balance_coeff

        # Message passing layers
        self.mp = torch.nn.ModuleList([
            GTVConv(in_channels if i == 0 else mp_channels, mp_channels,
                    eps=eps, delta_coeff=delta_coeff) for i in range(mp_layers)
        ])

        for layer in self.mp:
            torch.nn.init.kaiming_normal_(layer.weight)

        # MLP for producing cluster assignments
        self.pool = MLP(
            [mp_channels] +
            [mlp_hidden_channels
             for _ in range(mlp_hidden_layers)] + [n_clusters], act='relu',
            norm=None)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor):

        # Propagate node features
        for layer in self.mp:
            x = torch.nn.functional.elu(layer(x, edge_index, edge_weight))

        # Compute cluster assignment and obtain auxiliary losses
        s = self.pool(x)
        adj = utils.to_dense_adj(edge_index, edge_attr=edge_weight)
        _, _, tv_loss, bal_loss = dense_asymcheegercut_pool(x, adj, s)
        tv_loss *= self.totvar_coeff
        bal_loss *= self.balance_coeff

        return s.squeeze(0), tv_loss, bal_loss


model = Net(dataset.num_features, dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-7)


def train():
    model.train()
    optimizer.zero_grad()
    _, tv_loss, bal_loss = model(data.x, data.edge_index, data.edge_weight)
    loss = tv_loss + bal_loss
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test():
    model.eval()
    clust, _, _ = model(data.x, data.edge_index, data.edge_weight)
    return NMI(data.y.cpu(), clust.max(1)[1].cpu())


best_loss = 1
nmi_at_best_loss = 0
for epoch in range(500):
    train_loss = train()
    nmi = test()
    print(f"Epoch: {epoch:03d}, Loss: {train_loss:.4f}, NMI: {nmi:.3f}")
    if train_loss < best_loss:
        best_loss = train_loss
        nmi_at_best_loss = nmi

print(f"NMI: {nmi_at_best_loss:.3f}")
