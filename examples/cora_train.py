import argparse
import os.path as osp

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import torch_geometric.transforms as T
from torch_geometric import seed_everything
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import (
    PMLP,
    AGNNConv,
    ARMAConv,
    BatchNorm,
    DNAConv,
    GCN2Conv,
    GraphUNet,
    Linear,
    MixHopConv,
    SGConv,
    SplineConv,
    SuperGATConv,
    TAGConv,
)
from torch_geometric.typing import WITH_TORCH_SPLINE_CONV
from torch_geometric.utils import dropout_edge

if not WITH_TORCH_SPLINE_CONV:
    quit("This example requires 'torch-spline-conv'")

# define args
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter, )
parser.add_argument(
    '--dataset',
    type=str,
    default='Cora',
    choices=['Cora'],
    help='Dataset name.',
)
parser.add_argument(
    '--dataset_dir',
    type=str,
    default='./data',
    help='Root directory of dataset.',
)
parser.add_argument(
    '--gnn_choice',
    type=str,
    default='arma',
    choices=[
        'agnn', 'arma', 'mixhop', 'sgc', 'tagcn', 'graphunet', 'pmlp',
        'splinegnn', 'gcn2', 'super_gat', 'dna'
    ],
    help='Model name.',
)
parser.add_argument('-e', '--epochs', type=int, default=200)
parser.add_argument('--num_val', type=int, default=500)
parser.add_argument('--num_test', type=int, default=500)
parser.add_argument('--hidden_channels', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--wd', type=float, default=5e-5)
args = parser.parse_args()
seed_everything(123)

# detect device
if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

# define dataset, default train=140, val=500, test=1000
print(f'Training {args.dataset} with {args.gnn_choice} model.')

transform_lst = [
    T.RandomNodeSplit(num_val=args.num_val, num_test=args.num_test),
    T.TargetIndegree(),
    T.NormalizeFeatures(),
]
if args.gnn_choice == 'gcn2':
    transform_lst.extend([T.GCNNorm(), T.ToSparseTensor()])

root = osp.join(args.dataset_dir, args.dataset)
dataset = Planetoid(root, args.dataset, transform=T.Compose(transform_lst))
data = dataset[0].to(device)


# define models
class AGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_channels, hidden_channels)
        self.prop1 = AGNNConv(requires_grad=False)
        self.prop2 = AGNNConv(requires_grad=True)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.dropout(x, training=self.training)
        x = F.relu(self.lin1(x))
        x = self.prop1(x, edge_index)
        x = self.prop2(x, edge_index)
        x = F.dropout(x, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)


class ARMA(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = ARMAConv(in_channels, hidden_channels, num_stacks=3,
                              num_layers=2, shared_weights=True, dropout=0.25)
        self.conv2 = ARMAConv(hidden_channels, out_channels, num_stacks=3,
                              num_layers=2, shared_weights=True, dropout=0.25,
                              act=lambda x: x)

    def forward(self, x, edge_index):
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class MixHop(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = MixHopConv(in_channels, hidden_channels, powers=[0, 1, 2])
        self.norm1 = BatchNorm(3 * hidden_channels)
        self.conv2 = MixHopConv(3 * hidden_channels, hidden_channels,
                                powers=[0, 1, 2])
        self.norm2 = BatchNorm(3 * hidden_channels)
        self.conv3 = MixHopConv(3 * hidden_channels, hidden_channels,
                                powers=[0, 1, 2])
        self.norm3 = BatchNorm(3 * hidden_channels)
        self.lin = Linear(3 * hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.7, training=self.training)
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = F.dropout(x, p=0.9, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = F.dropout(x, p=0.9, training=self.training)
        x = self.conv3(x, edge_index)
        x = self.norm3(x)
        x = F.dropout(x, p=0.9, training=self.training)
        return F.log_softmax(self.lin(x), dim=1)


class SGC(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = SGConv(in_channels, out_channels, K=2, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        return F.log_softmax(x, dim=1)


class TAGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = TAGConv(in_channels, hidden_channels)
        self.conv2 = TAGConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GraphUNetModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_nodes):
        super().__init__()
        pool_ratios = [2000 / num_nodes, 0.5]
        self.unet = GraphUNet(in_channels, hidden_channels, out_channels,
                              depth=3, pool_ratios=pool_ratios)

    def forward(self, x, edge_index):
        edge_index, _ = dropout_edge(edge_index, p=0.2, force_undirected=True,
                                     training=self.training)
        x = F.dropout(x, p=0.92, training=self.training)

        x = self.unet(x, edge_index)
        return F.log_softmax(x, dim=1)


class SplineGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SplineConv(in_channels, hidden_channels, dim=1,
                                kernel_size=2)
        self.conv2 = SplineConv(hidden_channels, out_channels, dim=1,
                                kernel_size=2)

    def forward(self, x, edge_index, edge_attr=None):
        x = F.dropout(x, training=self.training)
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        return F.log_softmax(x, dim=1)


class GCN2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 alpha, theta, shared_weights=True, dropout=0.0):
        super().__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(in_channels, hidden_channels))
        self.lins.append(Linear(hidden_channels, out_channels))

        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                GCN2Conv(hidden_channels, alpha, theta, layer + 1,
                         shared_weights, normalize=False))

        self.dropout = dropout

    def forward(self, x, adj_t):
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()

        for conv in self.convs:
            x = F.dropout(x, self.dropout, training=self.training)
            x = conv(x, x_0, adj_t)
            x = x.relu()

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lins[1](x)

        return x.log_softmax(dim=-1)


class SuperGAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = SuperGATConv(in_channels, 8, heads=8, dropout=0.6,
                                  attention_type='MX', edge_sample_ratio=0.8,
                                  is_undirected=True)
        self.conv2 = SuperGATConv(8 * 8, out_channels, heads=8, concat=False,
                                  dropout=0.6, attention_type='MX',
                                  edge_sample_ratio=0.8, is_undirected=True)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        att_loss = self.conv1.get_attention_loss()
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, data.edge_index)
        att_loss += self.conv2.get_attention_loss()
        return F.log_softmax(x, dim=-1), att_loss


class DNA(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 heads=1, groups=1):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.lin1 = torch.nn.Linear(in_channels, hidden_channels)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(
                DNAConv(hidden_channels, heads, groups, dropout=0.8))
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index):
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x_all = x.view(-1, 1, self.hidden_channels)
        for conv in self.convs:
            x = F.relu(conv(x_all, edge_index))
            x = x.view(-1, 1, self.hidden_channels)
            x_all = torch.cat([x_all, x], dim=1)
        x = x_all[:, -1]
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return torch.log_softmax(x, dim=1)


def get_model(gnn_choice: str) -> torch.nn.Module:
    if gnn_choice == 'agnn':
        model = AGNN(
            dataset.num_features,
            args.hidden_channels,
            dataset.num_classes,
        )
    elif gnn_choice == 'arma':
        model = ARMA(
            dataset.num_features,
            args.hidden_channels,
            dataset.num_classes,
        )
    elif gnn_choice == 'mixhop':
        model = MixHop(
            dataset.num_features,
            args.hidden_channels,
            dataset.num_classes,
        )
    elif gnn_choice == 'sgc':
        model = SGC(
            dataset.num_features,
            dataset.num_classes,
        )
    elif gnn_choice == 'tagcn':
        model = TAGCN(
            dataset.num_features,
            args.hidden_channels,
            dataset.num_classes,
        )
    elif gnn_choice == 'graphunet':
        model = GraphUNetModel(
            dataset.num_features,
            args.hidden_channels,
            dataset.num_classes,
            data.num_nodes,
        )
    elif gnn_choice == 'pmlp':
        model = PMLP(
            in_channels=dataset.num_features,
            hidden_channels=args.hidden_channels,
            out_channels=dataset.num_classes,
            num_layers=2,
            dropout=0.5,
            norm=False,
        )
    elif gnn_choice == 'splinegnn':
        model = SplineGNN(
            dataset.num_features,
            args.hidden_channels,
            dataset.num_classes,
        )
    elif gnn_choice == 'gcn2':
        model = GCN2(
            dataset.num_features,
            args.hidden_channels,
            dataset.num_classes,
            num_layers=64,
            alpha=0.1,
            theta=0.5,
            shared_weights=True,
            dropout=0.6,
        )
    elif gnn_choice == 'super_gat':
        model = SuperGAT(
            dataset.num_features,
            dataset.num_classes,
        )
    elif gnn_choice == 'dna':
        model = DNA(
            dataset.num_features,
            args.hidden_channels,
            dataset.num_classes,
            num_layers=5,
            heads=8,
            groups=16,
        )
    else:
        raise ValueError(f'Unsupported model type: {gnn_choice}')
    return model


model = get_model(args.gnn_choice).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                             weight_decay=args.wd)


def train():
    model.train()
    optimizer.zero_grad()
    # forward
    if args.gnn_choice == 'splinegnn':
        out = model(data.x, data.edge_index, data.edge_attr)
    elif args.gnn_choice == 'gcn2':
        out = model(data.x, data.adj_t)
    elif args.gnn_choice == 'super_gat':
        out, att_loss = model(data.x, data.edge_index)
    else:
        out = model(data.x, data.edge_index)
    # loss
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    if args.gnn_choice == 'super_gat':
        loss += 4.0 * att_loss
    # backward
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def test():
    model.eval()
    accs = []
    # forward
    if args.gnn_choice == 'splinegnn':
        out = model(data.x, data.edge_index, data.edge_attr)
    elif args.gnn_choice == 'gcn2':
        out = model(data.x, data.adj_t)
    elif args.gnn_choice == 'super_gat':
        out, _ = model(data.x, data.edge_index)
    else:
        out = model(data.x, data.edge_index)
    # accuracy
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = out[mask].argmax(1)
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


writer = SummaryWriter()

best_val_acc = test_acc = 0
for epoch in range(args.epochs):
    train_loss = train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    print(
        f'Epoch: {epoch:03d}, Loss: {train_loss:.4f}, Train: {train_acc:.4f}, '
        f'Val: {best_val_acc:.4f}, Test: {test_acc:.4f}')

    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Accuracy/train', train_acc, epoch)
    writer.add_scalar('Accuracy/val', val_acc, epoch)
    writer.add_scalar('Accuracy/test', test_acc, epoch)
