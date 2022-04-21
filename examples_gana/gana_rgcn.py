import argparse
import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Entities
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.nn import RGCNConv, FastRGCNConv
from torch_geometric.loader import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('-k', '--num_layers', default=2, help="number of layers")
parser.add_argument('-epochs', '--epochs', default=2, help="number of epochs")
parser.add_argument('-hidden_channels', '--hidden_channels', default=16, help="number of channels in hidden layers")
args = parser.parse_args()

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Entities')

train_dataset = Entities(path, 'ppi', split='train')
val_dataset = Entities(path, 'ppi', split='val')
test_dataset = Entities(path, 'ppi', split='test')
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

print(train_dataset[0])
num_relations = train_dataset.num_relations
num_classes = train_dataset.num_classes
num_features = train_dataset.num_features
print(num_relations, num_classes, num_features)


class Net(torch.nn.Module):
    def __init__(self, num_layers, hidden_channels):
        super(Net, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            RGCNConv(train_dataset.num_features, hidden_channels,
                     num_relations, num_bases=30))
        for _ in range(num_layers - 2):
            self.convs.append(
                RGCNConv(hidden_channels, hidden_channels, num_relations,
                         num_bases=30))
        self.convs.append(RGCNConv(hidden_channels, num_classes, num_relations,
                                   num_bases=30))
        print(
            f"number of layers k:{len(self.convs)} hidden layersize :{hidden_channels}")

    def forward(self, data):
        x, edge_index, edge_type = data.x, data.edge_index, data.edge_type
        for conv in self.convs:
            x1 = conv(x, edge_index, edge_type)

            x = F.relu(x1)
            x = F.dropout(x, training=self.training)

        return F.log_softmax(x1, dim=1)


# class Net(torch.nn.Module):
#     def __init__(self, num_layers, hidden_channels):
#         super().__init__()
#         self.conv1 = RGCNConv(train_dataset.num_features, 16, num_relations,
#                               num_bases=30)
#         self.conv2 = RGCNConv(16, num_classes, num_relations,
#                               num_bases=30)

#     def forward(self, data):
#         x, edge_index, edge_type = data.x, data.edge_index, data.edge_type
#         x = F.relu(self.conv1(x, edge_index, edge_type))
#         x = self.conv2(x, edge_index, edge_type)
#         return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(int(args.num_layers), int(args.hidden_channels)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)


def train(train_loader):
    model.train()

    total_loss = total_examples = 0
    # print(f"{train_loader}")
    for data in train_loader:
        data = data.to(device)
        # print(f"data {data}")
        optimizer.zero_grad()
        # print(data.edge_index, data.edge_type)
        # print(f"batch size {data.x.size(0)}")
        loss = F.nll_loss(model(data), data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_nodes
        total_examples += data.num_nodes
    return total_loss / total_examples


@torch.no_grad()
def test(loader):
    model.eval()

    ys, preds = [], []
    for data in loader:
        ys.append(data.y)
        data = data.to(device)
        pred = model(data).argmax(dim=-1)
        preds.append(pred.cpu())

    y = torch.cat(ys, dim=0).to(torch.float)
    pred = torch.cat(preds, dim=0).to(torch.float)
    acc = pred.eq(y).to(torch.float).mean()
    return acc.item()


for epoch in range(1, int(args.epochs)):
    loss = train(train_loader)
    # train_acc = test(train_loader)
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
        df['true'] = (df['actual'] == df['predicted'])
        cm = confusion_matrix(y_val, y_pred)
        FP = cm.sum(axis=0) - np.diag(cm)
        FN = cm.sum(axis=1) - np.diag(cm)
        TP = np.diag(cm)
        TN = cm.sum() - (FP + FN + TP)
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        f1 = f1_score(data.y.cpu().numpy(), (out > 0).numpy(), average='micro')
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
