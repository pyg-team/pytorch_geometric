import os.path as osp
import tqdm

import torch
import torch.nn.functional as F
from torch.nn import Embedding, Linear

from torch_geometric.utils.convert import to_scipy_sparse_matrix
from torch_geometric.loader import LinkNeighborLoader
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv
from torch_geometric.datasets import Taobao

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

path = osp.join(osp.dirname(osp.realpath(__file__)), '../../data/Taobao')

dataset = Taobao(path)
data = dataset[0]

data['user'].x = torch.LongTensor(torch.arange(
    0, data['user'].num_nodes))
data['item'].x = torch.LongTensor(torch.arange(
    0, data['item'].num_nodes))

# Add a reverse ('item', 'rev_2', 'user') relation for message passing:
data = T.ToUndirected()(data)
del data['item', 'rev_2', 'user'].edge_label  # Remove "reverse" label.
del data[('user', '2', 'item')].edge_attr

# Perform a link-level split into training, validation, and test edges:
train_data, val_data, test_data = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    neg_sampling_ratio=0.0,
    edge_types=[('user', '2', 'item')],
    rev_edge_types=[('item', 'rev_2', 'user')],
)(data)

def to_u2i_mat(edge_index, u_num, i_num):
    # Convert bipartite edge_index format to matrix format
    u2imat = to_scipy_sparse_matrix(edge_index).tocsr()

    return u2imat[:u_num, :i_num]


def get_coocur_mat(train_mat, threshold):
    # Generate the co-occurrence matrix and top-k filtering
    A = train_mat.T @ train_mat
    A.setdiag(0)
    A = (A >= threshold).nonzero()
    A = torch.stack((torch.from_numpy(A[0]), torch.from_numpy(A[1])), dim=0)

    return A


u2i_mat = to_u2i_mat(train_data.edge_index_dict[('user', '2', 'item')],
                     train_data['user'].num_nodes,
                     train_data['item'].num_nodes)
i2i_edge_index = get_coocur_mat(u2i_mat, 3)

# Add the generated i2i graph for high-order information
train_data[('item', 'sims', 'item')].edge_index = i2i_edge_index
val_data[('item', 'sims', 'item')].edge_index = i2i_edge_index
test_data[('item', 'sims', 'item')].edge_index = i2i_edge_index


train_loader = LinkNeighborLoader(data=train_data,
                                  num_neighbors=[8, 4],
                                  edge_label_index=['user', '2', 'item'],
                                  neg_sampling_ratio=1.,
                                  batch_size=2048,
                                  num_workers=32,
                                  pin_memory=True,
)

val_loader = LinkNeighborLoader(data=val_data,
                                num_neighbors=[8, 4],
                                edge_label_index=['user', '2', 'item'],
                                neg_sampling_ratio=1.,
                                batch_size=2048,
                                num_workers=32,
                                pin_memory=True,
)

test_loader = LinkNeighborLoader(data=test_data,
                                 num_neighbors=[8, 4],
                                 edge_label_index=['user', '2', 'item'],
                                 neg_sampling_ratio=1.,
                                 batch_size=2048,
                                 num_workers=32,
                                 pin_memory=True,
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
            edge_index_dict[('item', 'sims', 'item')],
        ).relu()

        user_x = self.conv2(
            (x_dict['item'], x_dict['user']),
            edge_index_dict[('item', 'rev_2', 'user')],
        ).relu()

        user_x = self.conv3(
            (item_x, user_x),
            edge_index_dict[('item', 'rev_2', 'user')],
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
    def __init__(self, user_input_size, item_input_size, hidden_channels, out_channels):
        super().__init__()
        self.user_emb = Embedding(
            user_input_size, hidden_channels, device=device)
        self.item_emb = Embedding(
            item_input_size, hidden_channels, device=device)
        self.item_encoder = ItemGNNEncoder(hidden_channels, out_channels)
        self.user_encoder = UserGNNEncoder(hidden_channels, out_channels)
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = {}
        x_dict['user'] = self.user_emb(x_dict['user'])
        x_dict['item'] = self.item_emb(x_dict['item'])
        z_dict['item'] = self.item_encoder(
            x_dict['item'], edge_index_dict[('item', 'sims', 'item')])
        z_dict['user'] = self.user_encoder(x_dict, edge_index_dict)

        return self.decoder(z_dict['user'], z_dict['item'], edge_label_index)


model = Model(user_input_size=data['user'].num_nodes,
              item_input_size=data['item'].num_nodes,
              hidden_channels=64,
              out_channels=64)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)


def train():
    model.train()
    total_loss = 0
    for batch in tqdm.tqdm(train_loader):
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch.x_dict,
                     batch.edge_index_dict,
                     batch['user', 'item'].edge_label_index,
        )
        target = batch['user', 'item'].edge_label
        loss = F.binary_cross_entropy_with_logits(pred, target)
        loss.backward()
        optimizer.step()
        total_loss += loss

    return float(total_loss)


@torch.no_grad()
def test(loader):
    model.eval()
    total_acc, total_precision, total_recall, total_f1 = 0., 0., 0., 0.

    for batch in tqdm.tqdm(loader):
        batch = batch.to(device)
        out = model(batch.x_dict,
                    batch.edge_index_dict,
                    batch['user', 'item'].edge_label_index
        ).clamp(min=0, max=1).round().cpu()

        target = batch['user', 'item'].edge_label.round().cpu()
        acc = accuracy_score(target, out)
        precision = precision_score(target, out)
        recall = recall_score(target, out)
        f1 = f1_score(target, out)
        total_acc += acc
        total_precision += precision
        total_recall += recall
        total_f1 += f1

    return float(total_acc), float(total_precision), float(total_recall), float(total_f1)


for epoch in range(1, 51):
    loss = train()
    train_acc, train_precision, train_recall, train_f1 = test(train_loader)
    val_acc, val_precision, val_recall, val_f1 = test(val_loader)
    test_acc, test_precision, test_recall, test_f1 = test(test_loader)

    print(f'Epoch: {epoch:03d} | Loss: {loss:4f}')
    print(f'Eval Index: | Accuracy | Precision | Recall | F1_score')
    print(f'Train: {train_acc:.4f} | {train_precision:.4f} | {train_recall:.4f} \
          | {train_f1:.4f}')
    print(f'Val:   {val_acc:.4f} | {val_precision:.4f} | {val_recall:.4f} \
          | {val_f1:.4f}')
    print(f'Test:  {test_acc:.4f} | {test_precision:.4f} | {test_recall:.4f} \
          | {test_f1:.4f}')
