import os.path as osp

import torch

from torch_geometric.datasets import FB15k_237
from torch_geometric.nn import TransE

device = 'cuda' if torch.cuda.is_available() else 'cpu'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'FB15k')

train_data = FB15k_237(path, split='train')[0].to(device)
val_data = FB15k_237(path, split='val')[0].to(device)
test_data = FB15k_237(path, split='test')[0].to(device)

model = TransE(
    train_data.num_nodes,
    train_data.num_edge_types,
    hidden_channels=50,
).to(device)

loader = model.loader(
    head=train_data.edge_index[0],
    rel=train_data.edge_type,
    tail=train_data.edge_index[1],
    batch_size=10000,
    shuffle=True,
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train():
    model.train()
    total_loss = total_examples = 0
    for head, rel, tail in loader:
        optimizer.zero_grad()
        loss = model.loss(head, rel, tail)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * head.numel()
        total_examples += head.numel()
    return total_loss / total_examples


@torch.no_grad()
def test(data):
    model.eval()
    return model.test(
        head=data.edge_index[0],
        rel=data.edge_type,
        tail=data.edge_index[1],
        batch_size=20000,
        k=10,
    )


for epoch in range(1, 1001):
    loss = train()
    print(f'Epoch: {epoch:04d}, Loss: {loss:.4f}')
    if epoch % 100 == 0:
        rank, hits = test(val_data)
        print(f'Epoch: {epoch:04d}, Rank: {rank:.2f}, Hits: {hits:.4f}')
