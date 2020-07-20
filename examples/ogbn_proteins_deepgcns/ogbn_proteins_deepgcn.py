import torch
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm
from torch_scatter import scatter
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from torch_geometric.nn import GENConv
from torch_geometric.data import GraphSAINTNodeSampler

dataset = PygNodePropPredDataset('ogbn-proteins', root='../../data')
splitted_idx = dataset.get_idx_split()
data = dataset[0]
data.node_species = None
data.y = data.y.to(torch.float)

# Initialize features of nodes by aggregating edge features.
row, col = data.edge_index
data.x = scatter(data.edge_attr, col, 0, dim_size=data.num_nodes, reduce='add')

# Set split indices to masks.
for split in ['train', 'valid', 'test']:
    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[splitted_idx[split]] = True
    data[f'{split}_mask'] = mask


# Create `num_clusters` random subgraphs. This needs a special loader since
# each node should only be sampled once per epoch and subgraphs should not
# change across epochs.
class RandomClusterSampler(GraphSAINTNodeSampler):
    def __init__(self, data, num_clusters: int, **kwargs):
        self.cluster = torch.randint(num_clusters, (data.num_nodes, )).long()
        super(RandomClusterSampler,
              self).__init__(data, None, num_steps=num_clusters, **kwargs)

    def __getitem__(self, idx):
        node_idx = (self.cluster == idx).nonzero().view(-1)
        adj_t, edge_idx = self.adj_t.saint_subgraph(node_idx)
        edge_idx = self.adj_t.storage.value()[edge_idx]
        return node_idx, edge_idx, adj_t


train_loader = GraphSAINTNodeSampler(data, batch_size=data.num_nodes // 40,
                                     num_steps=40, num_workers=6)
test_loader = RandomClusterSampler(data, num_clusters=10, num_workers=6)


class DeeperGCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers):
        super(DeeperGCN, self).__init__()

        self.node_encoder = Linear(data.x.size(-1), hidden_channels)
        self.edge_encoder = Linear(data.edge_attr.size(-1), hidden_channels)

        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(
                GENConv(hidden_channels, hidden_channels, aggr='softmax',
                        t=1.0, learn_t=True, num_layers=2, norm='layer'))
            self.norms.append(
                LayerNorm(hidden_channels, elementwise_affine=True))

        self.lin = Linear(hidden_channels, data.y.size(-1))

    def forward(self, x, edge_index, edge_attr):
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            # Implement "res+" block:
            if i > 0 and i % 3 == 0 and x.requires_grad:
                h = checkpoint(conv, x, edge_index, edge_attr)
            else:
                h = conv(x, edge_index, edge_attr)

            h = norm(h)
            h = torch.relu_(h)
            h = F.dropout(h, p=0.1, training=self.training)
            x = x + h

        return self.lin(x)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DeeperGCN(hidden_channels=64, num_layers=28).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()
evaluator = Evaluator('ogbn-proteins')


def train(epoch):
    model.train()

    pbar = tqdm(total=len(train_loader))
    pbar.set_description(f'Training Epoch: {epoch:04d}')

    total_loss = total_examples = 0
    for data in train_loader:
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * int(data.train_mask.sum())
        total_examples += int(data.train_mask.sum())

        pbar.update(1)

    pbar.close()

    return total_loss / total_examples


@torch.no_grad()
def test():
    model.eval()

    y_true = {'train': [], 'valid': [], 'test': []}
    y_pred = {'train': [], 'valid': [], 'test': []}

    pbar = tqdm(total=len(test_loader))
    pbar.set_description(f'Evaluating Epoch: {epoch:04d}')

    for data in test_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr)

        for split in y_true.keys():
            mask = data[f'{split}_mask']
            y_true[split].append(data.y[mask].cpu())
            y_pred[split].append(out[mask].cpu())

        pbar.update(1)

    pbar.close()

    train_rocauc = evaluator.eval({
        'y_true': torch.cat(y_true['train'], dim=0),
        'y_pred': torch.cat(y_pred['train'], dim=0),
    })['rocauc']

    valid_rocauc = evaluator.eval({
        'y_true': torch.cat(y_true['valid'], dim=0),
        'y_pred': torch.cat(y_pred['valid'], dim=0),
    })['rocauc']

    test_rocauc = evaluator.eval({
        'y_true': torch.cat(y_true['test'], dim=0),
        'y_pred': torch.cat(y_pred['test'], dim=0),
    })['rocauc']

    return train_rocauc, valid_rocauc, test_rocauc


for epoch in range(1, 1001):
    loss = train(epoch)
    train_rocauc, valid_rocauc, test_rocauc = test()
    print(f'Loss: {loss:.4f}, Train: {train_rocauc:.4f}, '
          f'Val: {valid_rocauc:.4f}, Test: {test_rocauc:.4f}')
