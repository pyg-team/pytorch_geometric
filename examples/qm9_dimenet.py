import torch
import os.path as osp
from torch_geometric.datasets import QM9
from torch_geometric.data import DataLoader
from torch_geometric.nn import DimeNet
from torch_sparse import SparseTensor

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'QM9')
# dataset = QM9(path).shuffle()
dataset = QM9(path)
train_dataset = dataset[:110000]
valid_dataset = dataset[110000:120000]
test_dataset = dataset[120000:]
print(dataset)
print(train_dataset)
print(valid_dataset)
print(test_dataset)
# transform = T.Compose([MyTransform(), Complete(), T.Distance(norm=False)])

data = dataset[0]
print(data)

cutoff = 5.0

dist = (data.pos.view(-1, 1, 3) - data.pos.view(1, -1, 3)).norm(dim=-1)
dist.fill_diagonal_(float('inf'))
mask = dist <= cutoff
edge_index = mask.nonzero().t()

in_channels = 13
model = DimeNet(in_channels, hidden_channels=128, out_channels=1, num_blocks=6,
                num_bilinear=8, num_spherical=7, num_radial=6, cutoff=cutoff)

model(data.x, data.pos, edge_index)

# batch_size = 32
# num_steps = 3000000
# num_epochs = int(num_steps / (len(train_dataset) / batch_size))
# train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

# def train(epoch):
#     model.train()
#     loss_all = 0

#     for data in train_loader:
#         data = data.to(device)
#         optimizer.zero_grad()
#         loss = F.mse_loss(model(data), data.y)
#         loss.backward()
#         loss_all += loss.item() * data.num_graphs
#         optimizer.step()
#     return loss_all / len(train_loader.dataset)
