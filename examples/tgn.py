import os.path as osp

import torch
from torch.nn import Linear
from torch_sparse import SparseTensor
from sklearn.metrics import average_precision_score, roc_auc_score

from torch_geometric.datasets import JODIEDataset
from torch_geometric.nn import TGN, SAGEConv
from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'JODIE')
dataset = JODIEDataset(path, name='wikipedia')
data = dataset[0].to(device)
min_dst_id = min(data.dst.unique())
max_dst_id = max(data.dst.unique())


class NeighborSampler(object):
    def __init__(self, data, size):
        self.size = size
        self.device = data.src.device
        self.adj = SparseTensor(row=data.src.cpu(), col=data.dst.cpu(),
                                value=torch.arange(data.num_events),
                                sparse_sizes=(data.num_nodes, data.num_nodes))

    def __call__(self, n_id, threshold):
        _, _, value = self.adj.coo()
        mask = value < threshold
        adj = self.adj.masked_select_nnz(mask, layout='coo').set_value_(None)
        if adj.numel() == 0:
            adj = adj.sparse_resize([n_id.numel(), n_id.numel()])
        else:
            adj, n_id = adj.sample_adj(n_id.cpu(), num_neighbors=self.size)
        return adj.to(self.device), n_id.to(self.device)


sampler = NeighborSampler(data, size=10)
train_data, val_data, test_data = data.train_val_test_split(
    val_ratio=0.15, test_ratio=0.15)


class GraphEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphEmbedding, self).__init__()
        self.conv = SAGEConv(in_channels, out_channels)

    def forward(self, x, adj_t):
        if adj_t.nnz() > 0:
            x = self.conv((x, x[:adj_t.size(0)]), adj_t)
        else:
            x = self.conv.lin_r(x)
        return x.relu()


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels):
        super(LinkPredictor, self).__init__()
        self.lin_src = Linear(in_channels, in_channels)
        self.lin_dst = Linear(in_channels, in_channels)
        self.lin_end = Linear(in_channels, 1)

    def forward(self, z_src, z_dst):
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = h.relu()
        return self.lin_end(h)


model = TGN(
    data.num_nodes, data.msg.size(-1), memory_dim=100, time_dim=100,
    message_module=IdentityMessage(raw_msg_dim=data.msg.size(-1),
                                   memory_dim=100, time_dim=100),
    aggregator_module=LastAggregator()).to(device)
gnn = GraphEmbedding(in_channels=100, out_channels=100).to(device)
link_pred = LinkPredictor(in_channels=100).to(device)

optimizer = torch.optim.Adam(
    list(model.parameters()) + list(gnn.parameters()) +
    list(link_pred.parameters()), lr=0.0001)
criterion = torch.nn.BCEWithLogitsLoss()

assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)


def train():
    model.train()
    model.reset_state()
    link_pred.train()

    total_loss = current_event_id = 0
    for batch in train_data.seq_batches(batch_size=200):
        optimizer.zero_grad()

        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
        neg_dst = torch.randint(min_dst_id, max_dst_id + 1, (src.size(0), ),
                                dtype=torch.long, device=device)

        n_id = torch.cat([src, pos_dst, neg_dst]).unique()
        # adj_t, n_id = sampler(n_id, threshold=current_event_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # Sample 1-hop neighborhood.

        z, _ = model(n_id, t)  # Get memory.
        # z = gnn(z, adj_t)  # Embed memory via graph convolution.

        pos_out = link_pred(z[assoc[src]], z[assoc[pos_dst]])
        neg_out = link_pred(z[assoc[src]], z[assoc[neg_dst]])

        loss = criterion(pos_out, torch.ones_like(pos_out))
        loss += criterion(neg_out, torch.zeros_like(neg_out))

        model.update_state(src, pos_dst, t, msg)

        loss.backward()
        optimizer.step()
        model.detach_memory()
        total_loss += float(loss) * batch.num_events
        current_event_id += batch.num_events

    return total_loss / train_data.num_events


@torch.no_grad()
def test(inference_data):
    model.eval()
    link_pred.eval()
    torch.manual_seed(12345)  # Ensure deterministic sampling across epochs.

    aps, aucs = [], []
    current_event_id = train_data.num_events
    for batch in inference_data.seq_batches(batch_size=200):
        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
        neg_dst = torch.randint(min_dst_id, max_dst_id + 1, (src.size(0), ),
                                dtype=torch.long, device=device)

        n_id = torch.cat([src, pos_dst, neg_dst]).unique()
        # adj_t, n_id = sampler(n_id, threshold=current_event_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        z, _ = model(n_id, t, train=False)
        # z = gnn(z, adj_t)  # Embed memory via graph convolution.

        pos_out = link_pred(z[assoc[src]], z[assoc[pos_dst]])
        neg_out = link_pred(z[assoc[src]], z[assoc[neg_dst]])

        y_pred = torch.cat([pos_out, neg_out], dim=0).sigmoid().cpu()
        y_true = torch.cat(
            [torch.ones(pos_out.size(0)),
             torch.zeros(neg_out.size(0))], dim=0)

        aps.append(average_precision_score(y_true, y_pred))
        aucs.append(roc_auc_score(y_true, y_pred))

        model.update_state(src, pos_dst, t, msg, train=False)
        current_event_id += batch.num_events

    return float(torch.tensor(aps).mean()), float(torch.tensor(aucs).mean())


for epoch in range(1, 51):
    loss = train()
    print(f'  Epoch: {epoch:02d}, Loss: {loss:.4f}')
    val_ap, val_auc = test(val_data)
    test_ap, test_auc = test(test_data)
    print(f' Val AP: {val_ap:.4f},  Val AUC: {val_auc:.4f}')
    print(f'Test AP: {test_ap:.4f}, Test AUC: {test_auc:.4f}')
