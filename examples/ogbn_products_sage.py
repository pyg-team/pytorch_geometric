# Reaches around 0.7870 ± 0.0036 test accuracy.

import os.path as osp

import torch
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from tqdm import tqdm

from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'products')
dataset = PygNodePropPredDataset('ogbn-products', root)
split_idx = dataset.get_idx_split()
evaluator = Evaluator(name='ogbn-products')
data = dataset[0].to(device)
train_idx = split_idx['train']
train_loader = NeighborLoader(data, input_nodes=train_idx,
                               num_neighbors=[15, 10, 5], batch_size=1024,
                               shuffle=True, num_workers=12)
subgraph_loader = NeighborLoader(data, input_nodes=None, num_neighbors=[-1],
                                  batch_size=4096, shuffle=False,
                                  num_workers=12)

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, batch):
        x, edge_index = batch.x, batch.edge_index
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)

    def inference(self, x_all):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        total_edges = 0
        for i in range(self.num_layers):
            xs = []
            for batch in subgraph_loader:
                edge_index = batch.edge_index
                total_edges += edge_index.size(1)
                x = x_all[batch.n_id[batch.batch_size]].to(device)
                x = self.convs[i](x, edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all


model = SAGE(dataset.num_features, 256, dataset.num_classes, num_layers=3)
model = model.to(device)

x = data.x
y = data.y.squeeze()


def train(epoch):
    model.train()

    pbar = tqdm(total=train_idx.size(0))
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = 0
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch)
        o = out[:batch_size]
        y_true = y[batch.n_id[:batch_size]]
        loss = F.nll_loss(o, y_true)
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        total_correct += int(o.argmax(dim=-1).eq(y_true).sum())
        pbar.update(batch_size)

    pbar.close()

    loss = total_loss / len(train_loader)
    approx_acc = total_correct / train_idx.size(0)

    return loss, approx_acc


@torch.no_grad()
def test():
    model.eval()

    out = model.inference(x)

    y_true = y.cpu().unsqueeze(-1)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    val_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, val_acc, test_acc


test_accs = []
for run in range(1, 11):
    print('')
    print(f'Run {run:02d}:')
    print('')

    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    best_val_acc = final_test_acc = 0
    for epoch in range(1, 21):
        loss, acc = train(epoch)
        print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}')

        if epoch > 5:
            train_acc, val_acc, test_acc = test()
            print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
                  f'Test: {test_acc:.4f}')

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                final_test_acc = test_acc
    test_accs.append(final_test_acc)

test_acc = torch.tensor(test_accs)
print('============================')
print(f'Final Test: {test_acc.mean():.4f} ± {test_acc.std():.4f}')
