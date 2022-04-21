import os.path as osp
import argparse
import csv
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import PPI
from sklearn.metrics import f1_score
from torch_geometric.nn import SAGEConv


parser = argparse.ArgumentParser()
parser.add_argument('-k', '--num_layers', default=2, help="number of layers")
parser.add_argument('-epochs', '--epochs', default=2, help="number of epochs")
parser.add_argument('-hidden_channels', '--hidden_channels', default=16, help="number of channels in hidden layers")
parser.add_argument('--use_gdc', action='store_true',
                    help='Use GDC preprocessing.')
args = parser.parse_args()

# dataset = 'Cora'
# path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
# dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
# data = dataset[0]

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'PPI_GANA')
train_dataset = PPI(path, split='train')
val_dataset = PPI(path, split='val')
test_dataset = PPI(path, split='test')
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# if args.use_gdc:
#     gdc = T.GDC(self_loop_weight=1, normalization_in='sym',
#                 normalization_out='col',
#                 diffusion_kwargs=dict(method='ppr', alpha=0.05),
#                 sparsification_kwargs=dict(method='topk', k=128,
#                                            dim=0), exact=True)
#     data = gdc(data)


class Net(torch.nn.Module):
    def __init__(self, num_layers, hidden_channels):
        super(Net, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            SAGEConv(train_dataset.num_features, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, train_dataset.num_classes))

    def forward(self, x, edge_index):
        for conv in self.convs:
            x1 = conv(x, edge_index)
            x = F.relu(x1)
            x = F.dropout(x, training=self.training)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(int(args.num_layers), int(args.hidden_channels)).to(device)
loss_op = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)


def train():
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = loss_op(model(data.x, data.edge_index), data.y)
        total_loss += loss.item() * data.num_graphs
        loss.backward()
        optimizer.step()
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader):
    model.eval()

    ys, preds = [], []
    for data in loader:
        ys.append(data.y)
        out = model(data.x.to(device), data.edge_index.to(device))
        preds.append((out > 0).float().cpu())

    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
    return f1_score(y, pred, average='micro') if pred.sum() > 0 else 0


@torch.no_grad()
def debug(loader):
    model.eval()
    merged = []
    ys, preds = [], []
    for data in loader:
        ys.append(data.y)
        out = model(data.x.to(device), data.edge_index.to(device))
        preds.append((out > 0).float().cpu())
        debug = torch.cat((data.x.to(device), data.y.to(device), (out > 0)), dim=1)
        merged.append(debug.cpu())
    with open("debug_sage_result.csv", 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(merged)

    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
    return f1_score(y, pred, average='micro') if pred.sum() > 0 else 0

for epoch in range(1, int(args.epochs)):
    loss = train()
    val_f1 = test(val_loader)
    test_f1 = test(test_loader)
    print('Epoch: {:02d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}'.format(
        epoch, loss, val_f1, test_f1))


@torch.no_grad()
def debug(loader):
    import os
    import json
    import matplotlib.pylab as plt
    import time
    import pandas as pd
    import numpy as np
    from sklearn.metrics import f1_score, confusion_matrix
    from torch_geometric.utils import to_networkx
    import networkx as nx

    model.eval()
    ys, preds = [], []
    assert os.path.exists(
        '../data/PPI_GANA/raw/valid_name_map.json'), f'no map file found '
    with open('../data/PPI_GANA/raw/valid_name_map.json', "r") as f:
        map_data = json.load(f)
    x_names = [map_data[str(i)] for i in range(len(map_data))]
    count = 0

    result_dir = f"results/{os.path.basename(__file__).split('.')[0]}/{args.num_layers}_{args.hidden_channels}"
    print(result_dir)
    os.makedirs(result_dir, exist_ok=True)

    for data in loader:
        plt.figure(figsize=(12, 12))
        ys.append(data.y)
        start = time.time()
        out = model(data.x.to(device), data.edge_index.to(device))
        end = time.time()
        preds.append((out > 0).float().cpu())
        eles, x_names = x_names[:len(data.y)], x_names[len(data.y):]
        y_val = [row.index(1) for row in data.y.tolist()]
        y_pred = [row.index(True) for row in (out > 0).tolist()]
        df = pd.DataFrame({'actual': y_val, 'predicted': y_pred})
        df['true'] = (df['actual'] != df['predicted'])
        cm = confusion_matrix(y_val, y_pred)
        FP = cm.sum(axis=0) - np.diag(cm)
        FN = cm.sum(axis=1) - np.diag(cm)
        TP = np.diag(cm)
        TN = cm.sum() - (FP + FN + TP)
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        f1 = f1_score(data.y.cpu().numpy(), (out > 0).cpu().numpy(), average='micro')
        G = to_networkx(data, node_attrs=['y'], to_undirected=True)
        mapping = {k: i for k, i in enumerate(eles)}
        incorrect_nodes = [v for k, v in mapping.items() if df['true'][k]]
        G = nx.relabel_nodes(G, mapping)
        node_kwargs = {}
        node_kwargs['node_size'] = 800
        node_kwargs['cmap'] = 'cool'
        label_kwargs = {}
        label_kwargs['font_size'] = 10
        ax = plt.gca()
        pos = nx.spring_layout(G, seed=1)
        for source, target, _ in G.edges(data=True):
            ax.annotate(
                '', xy=pos[target], xycoords='data', xytext=pos[source],
                textcoords='data', arrowprops=dict(
                    arrowstyle="-",
                ))

        nx.draw_networkx_nodes(G, pos, node_color=y_pred)
        nx.draw_networkx_nodes(
            G, pos, nodelist=incorrect_nodes, node_color="tab:red")
        nx.draw_networkx_labels(G, pos, **label_kwargs)
        df['name'] = eles
        plt.title(
            f"f1 score: {format(f1,'.2')} TPR: {TPR} FPR: {FPR} time: {format(1000*(end - start), '.4')}")
        plt.savefig(
            f"{result_dir}/{str(count)}.png")
        df.to_csv(
            f"{result_dir}/{str(count)}.csv")
        count += 1
        if count == 5:
            break


debug(val_loader)
