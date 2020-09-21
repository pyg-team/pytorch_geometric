import os.path as osp

import argparse
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GAE, VGAE
from torch_geometric.utils import train_test_split_edges

torch.manual_seed(12345)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='GAE')
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--epochs', type=int, default=400)
args = parser.parse_args()
assert args.model in ['GAE', 'LinearGAE', 'VGAE', 'LinearVGAE']
assert args.dataset in ['Cora', 'CiteSeer', 'PubMed']
kwargs = {'GAE': GAE, 'LinearGAE': GAE, 'VGAE': VGAE, 'LinearVGAE': VGAE}

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data',
                args.dataset)
dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
data = dataset[0]


# 2-layer GCN encoder from standard graph AE and VAE models: https://arxiv.org/abs/1611.07308
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
        if args.model in ['GAE']:
            self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True)
        elif args.model in ['VGAE']:
            self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=True)
            self.conv_logstd = GCNConv(2 * out_channels, out_channels,
                                       cached=True)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        if args.model in ['GAE']:
            return self.conv2(x, edge_index)
        elif args.model in ['VGAE']:
            return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

          
# Linear encoder from linear graph AE and VAE models: https://arxiv.org/abs/1910.00942
class LinearEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearEncoder, self).__init__()
        if args.model in ['LinearGAE']:
            self.conv1 = GCNConv(in_channels, out_channels, cached=True)
        elif args.model in ['LinearVGAE']:
            self.conv_mu = GCNConv(in_channels, out_channels, cached=True)
            self.conv_logstd = GCNConv(in_channels, out_channels,
                                       cached=True)

    def forward(self, x, edge_index):
        if args.model in ['LinearGAE']:
            return self.conv1(x, edge_index)
        elif args.model in ['LinearVGAE']:
            return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


channels = 16
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if args.model in ['LinearGAE','LinearVGAE']:
    model = kwargs[args.model](LinearEncoder(dataset.num_features, channels)).to(dev)
else:
    model = kwargs[args.model](GCNEncoder(dataset.num_features, channels)).to(dev)
data.train_mask = data.val_mask = data.test_mask = data.y = None
data = train_test_split_edges(data)
x, train_pos_edge_index = data.x.to(dev), data.train_pos_edge_index.to(dev)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(x, train_pos_edge_index)
    loss = model.recon_loss(z, train_pos_edge_index)
    if args.model in ['VGAE','LinearVGAE']:
        loss = loss + (1 / data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()


def test(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index)
    return model.test(z, pos_edge_index, neg_edge_index)


for epoch in range(1, args.epochs + 1):
    train()
    auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index)
    print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))
