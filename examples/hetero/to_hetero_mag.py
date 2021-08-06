import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import ReLU
import torch_geometric.transforms as T
from torch_geometric.nn import Linear, Sequential, SAGEConv, to_hetero
from torch_geometric.data import HGTSampler
from torch_geometric.datasets import OGB_MAG

path = osp.join(osp.dirname(osp.realpath(__file__)), '../../data/OGB')
transform = T.Compose([T.ToUndirected(), T.ToSparseTensor()])
dataset = OGB_MAG(path, pre_transform=transform)
data = dataset[0]

# Pre-fill missing node features:
for src, rel, dst in [('paper', 'inv_writes', 'author'),
                      ('paper', 'has_topic', 'field_of_study'),
                      ('author', 'affiliated_with', 'institution')]:
    data[dst].x = data[rel].adj_t.matmul(data[src].x, 'mean')

train_loader = HGTSampler(data, num_samples=[1024] * 4, batch_size=1024,
                          num_workers=6, persistent_workers=True, shuffle=True,
                          input_nodes=('paper', data['paper'].train_mask))
val_loader = HGTSampler(data, num_samples=[1024] * 4, batch_size=1024,
                        num_workers=6, persistent_workers=True,
                        input_nodes=('paper', data['paper'].val_mask))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Sequential('x, adj', [
    (SAGEConv((-1, -1), 64), 'x, adj -> x'),
    ReLU(inplace=True),
    (SAGEConv((-1, -1), 64), 'x, adj -> x'),
    ReLU(inplace=True),
    (Linear(-1, dataset.num_classes), 'x -> x'),
])
model = to_hetero(model, data.metadata(), aggr='sum').to(device)


def train():
    model.train()

    total_examples = total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        batch_size = batch['paper'].batch_size
        out = model(batch.x_dict, batch.adj_t_dict)['paper'][:batch_size]
        loss = F.cross_entropy(out, batch['paper'].y[:batch_size])
        loss.backward()
        optimizer.step()

        total_examples += batch_size
        total_loss += float(loss) * batch_size

    return total_loss / total_examples


@torch.no_grad()
def test(loader):
    model.eval()

    total_examples = total_correct = 0
    for batch in val_loader:
        batch = batch.to(device)
        batch_size = batch['paper'].batch_size
        out = model(batch.x_dict, batch.adj_t_dict)['paper'][:batch_size]
        pred = out.argmax(dim=-1)

        total_examples += batch_size
        total_correct += int((pred == batch['paper'].y[:batch_size]).sum())

    return total_correct / total_examples


val_acc = test(val_loader)  # Initialize parameters.
print('#Params:', sum([p.numel() for p in model.parameters()]))  # Ensure.
print(f'Epoch: 00, Val: {val_acc:.4f}')
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(1, 21):
    loss = train()
    train_acc = test(train_loader)
    val_acc = test(val_loader)
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
          f'Val: {val_acc:.4f}')
