import os.path as osp

import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.datasets import JODIEDataset
from torch_geometric.nn import TGN
from torch_geometric.nn.models.tgn import (IdentityMessage, LastAggregator,
                                           IdentityEmbedding)

# from torch_geometric.nn.models.tgn import (NeighborSampler,
#                                            TemporalGraphMeanGNN)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'JODIE')
dataset = JODIEDataset(path, name='wikipedia')
data = dataset[0].to(device)
train_data, val_data, test_data = data.train_val_test_split(
    val_ratio=0.15, test_ratio=0.15)

# sampler = NeighborSampler(src=data.src, dst=data.dst, t=data.t, raw_msg=data.x,
#                           sizes=[10])
# GNN = TemporalGraphMeanGNN(memory_dim=100, raw_msg_dim=172, time_dim=100,
#                            out_channels=256, sampler=sampler)

model = TGN(
    data, memory_dim=100, time_dim=100,
    message_module=IdentityMessage(memory_dim=100, raw_msg_dim=172,
                                   time_dim=100),
    aggregator_module=LastAggregator(),
    embedding_module=IdentityEmbedding(memory_dim=100))
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = torch.nn.BCEWithLogitsLoss()


def train():
    model.train()
    model.reset_state()

    total_loss = 0
    backprop_every = 1
    for i, batch in enumerate(train_data.seq_batches(batch_size=5)):

        # if i % backprop_every == 0:
        #     loss = 0
        #     optimizer.zero_grad()

        src, pos_dst, t, x = batch.src, batch.dst, batch.t, batch.x
        neg_dst = torch.randint(0, train_data.num_nodes, (src.size(0), ),
                                dtype=torch.long, device=device)

        model(src, pos_dst, neg_dst, t, x)

        if i == 2:
            raise NotImplementedError
        # pos_out, neg_out = model(src, pos_dst, neg_dst, t, x)
        # loss += criterion(pos_out, torch.ones_like(pos_out))
        # loss += criterion(neg_out, torch.zeros_like(neg_out))

        # total_loss += float(loss) * batch.num_events

        # if (i + 1) % backprop_every == 0:
        #     loss = loss / backprop_every
        #     loss.backward()
        #     optimizer.step()
        #     model.detach_memory()

    return total_loss / train_data.num_events


@torch.no_grad()
def test(inference_data):
    model.eval()
    torch.manual_seed(12345)  # Ensure deterministic sampling across epochs.

    aps, aucs = [], []
    for batch in inference_data.seq_batches(batch_size=200):
        src, pos_dst, t, x = batch.src, batch.dst, batch.t, batch.x
        neg_dst = torch.randint(0, train_data.num_nodes, (src.size(0), ),
                                dtype=torch.long, device=device)

        pos_out, neg_out = model(src, pos_dst, neg_dst, t, x)

        y_pred = torch.cat([pos_out, neg_out], dim=0).sigmoid().cpu()
        y_true = torch.cat(
            [torch.ones(pos_out.size(0)),
             torch.zeros(neg_out.size(0))], dim=0)

        aps.append(average_precision_score(y_true, y_pred))
        aucs.append(roc_auc_score(y_true, y_pred))

    return float(torch.tensor(aps).mean()), float(torch.tensor(aucs).mean())


for epoch in range(1, 51):
    loss = train()
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
    val_ap, val_auc = test(val_data)
    test_ap, test_auc = test(test_data)
    print(f' Val AP: {val_ap:.4f},  Val AUC: {val_auc:.4f}')
    print(f'Test AP: {val_ap:.4f}, Test AUC: {test_auc:.4f}')
