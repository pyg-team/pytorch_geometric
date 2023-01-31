# An implementation of unsupervised bipartite GraphSAGE using the Alibaba
# Taobao dataset.
import os.path as osp

import torch
import torch.nn.functional as F
import tqdm
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch.nn import Embedding, Linear

import torch_geometric.transforms as T
from torch_geometric.datasets import Taobao
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import SAGEConv
from torch_geometric.utils.convert import to_scipy_sparse_matrix

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
path = osp.join(osp.dirname(osp.realpath(__file__)), '../../data/Taobao')

dataset = Taobao(path)
data = dataset[0]

data['user'].x = torch.arange(0, data['user'].num_nodes)
data['item'].x = torch.arange(0, data['item'].num_nodes)

# Only consider user<>item relationships for simplicity:
del data['category']
del data['item', 'category']
del data['user', 'item'].time
del data['user', 'item'].behavior

# Add a reverse ('item', 'rev_to', 'user') relation for message passing:
data = T.ToUndirected()(data)

# Perform a link-level split into training, validation, and test edges:
print('Computing data splits...')
train_data, val_data, test_data = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    neg_sampling_ratio=1.0,
    add_negative_train_samples=False,
    edge_types=[('user', 'to', 'item')],
    rev_edge_types=[('item', 'rev_to', 'user')],
)(data)
print('Done!')

# Compute sparsified item<>item relationships through users:
print('Computing item<>item relationships...')
mat = to_scipy_sparse_matrix(data['user', 'item'].edge_index).tocsr()
mat = mat[:data['user'].num_nodes, :data['item'].num_nodes]
comat = mat.T @ mat
comat.setdiag(0)
comat = comat >= 3.
comat = comat.tocoo()
row = torch.from_numpy(comat.row).to(torch.long)
col = torch.from_numpy(comat.col).to(torch.long)
item_to_item_edge_index = torch.stack([row, col], dim=0)

# Add the generated item<>item relationships for high-order information:
train_data['item', 'item'].edge_index = item_to_item_edge_index
val_data['item', 'item'].edge_index = item_to_item_edge_index
test_data['item', 'item'].edge_index = item_to_item_edge_index
print('Done!')

train_loader = LinkNeighborLoader(
    data=train_data,
    num_neighbors=[8, 4],
    edge_label_index=('user', 'to', 'item'),
    neg_sampling='binary',
    batch_size=2048,
    shuffle=True,
    num_workers=16,
    drop_last=True,
)

val_loader = LinkNeighborLoader(
    data=val_data,
    num_neighbors=[8, 4],
    edge_label_index=(
        ('user', 'to', 'item'),
        val_data[('user', 'to', 'item')].edge_label_index,
    ),
    edge_label=val_data[('user', 'to', 'item')].edge_label,
    batch_size=2048,
    shuffle=False,
    num_workers=16,
)

test_loader = LinkNeighborLoader(
    data=test_data,
    num_neighbors=[8, 4],
    edge_label_index=(
        ('user', 'to', 'item'),
        test_data[('user', 'to', 'item')].edge_label_index,
    ),
    edge_label=test_data[('user', 'to', 'item')].edge_label,
    batch_size=2048,
    shuffle=False,
    num_workers=16,
)


class ItemGNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(-1, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return self.lin(x)


class UserGNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels)
        self.conv3 = SAGEConv((-1, -1), hidden_channels)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        item_x = self.conv1(
            x_dict['item'],
            edge_index_dict[('item', 'to', 'item')],
        ).relu()

        user_x = self.conv2(
            (x_dict['item'], x_dict['user']),
            edge_index_dict[('item', 'rev_to', 'user')],
        ).relu()

        user_x = self.conv3(
            (item_x, user_x),
            edge_index_dict[('item', 'rev_to', 'user')],
        ).relu()

        return self.lin(user_x)


class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, z_src, z_dst, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_src[row], z_dst[col]], dim=-1)

        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)


class Model(torch.nn.Module):
    def __init__(self, num_users, num_items, hidden_channels, out_channels):
        super().__init__()
        self.user_emb = Embedding(num_users, hidden_channels, device=device)
        self.item_emb = Embedding(num_items, hidden_channels, device=device)
        self.item_encoder = ItemGNNEncoder(hidden_channels, out_channels)
        self.user_encoder = UserGNNEncoder(hidden_channels, out_channels)
        self.decoder = EdgeDecoder(out_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = {}
        x_dict['user'] = self.user_emb(x_dict['user'])
        x_dict['item'] = self.item_emb(x_dict['item'])
        z_dict['item'] = self.item_encoder(
            x_dict['item'],
            edge_index_dict[('item', 'to', 'item')],
        )
        z_dict['user'] = self.user_encoder(x_dict, edge_index_dict)

        return self.decoder(z_dict['user'], z_dict['item'], edge_label_index)


model = Model(
    num_users=data['user'].num_nodes,
    num_items=data['item'].num_nodes,
    hidden_channels=64,
    out_channels=64,
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train():
    model.train()

    total_loss = total_examples = 0
    for batch in tqdm.tqdm(train_loader):
        batch = batch.to(device)
        optimizer.zero_grad()

        pred = model(
            batch.x_dict,
            batch.edge_index_dict,
            batch['user', 'item'].edge_label_index,
        )
        loss = F.binary_cross_entropy_with_logits(
            pred, batch['user', 'item'].edge_label)

        loss.backward()
        optimizer.step()
        total_loss += float(loss)
        total_examples += pred.numel()

    return total_loss / total_examples


@torch.no_grad()
def test(loader):
    model.eval()

    preds, targets = [], []
    for batch in tqdm.tqdm(loader):
        batch = batch.to(device)

        pred = model(
            batch.x_dict,
            batch.edge_index_dict,
            batch['user', 'item'].edge_label_index,
        ).sigmoid().view(-1).cpu()
        target = batch['user', 'item'].edge_label.long().cpu()

        preds.append(pred)
        targets.append(target)

    pred = torch.cat(preds, dim=0).numpy()
    target = torch.cat(targets, dim=0).numpy()

    pred = pred > 0.5
    acc = accuracy_score(target, pred)
    prec = precision_score(target, pred)
    rec = recall_score(target, pred)
    f1 = f1_score(target, pred)

    return acc, prec, rec, f1


for epoch in range(1, 21):
    loss = train()
    val_acc, val_prec, val_rec, val_f1 = test(val_loader)
    test_acc, test_prec, test_rec, test_f1 = test(test_loader)

    print(f'Epoch: {epoch:03d}, Loss: {loss:4f}')
    print(f'Val Acc: {val_acc:.4f}, Val Precision {val_prec:.4f}, '
          f'Val Recall {val_rec:.4f}, Val F1 {val_f1:.4f}')
    print(f'Test Acc: {test_acc:.4f}, Test Precision {test_prec:.4f}, '
          f'Test Recall {test_rec:.4f}, Test F1 {test_f1:.4f}')
