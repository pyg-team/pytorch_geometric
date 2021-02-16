import torch
import os.path as osp

import torch.nn.functional as F
from torch_geometric.datasets import ZINC
from torch_geometric.data import DataLoader

ind2ptr = torch.ops.torch_sparse.ind2ptr  # (row, self._sparse_sizes[0])

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'ZINC')
train_dataset = ZINC(path, subset=True, split='train')
val_dataset = ZINC(path, subset=True, split='val')
test_dataset = ZINC(path, subset=True, split='test')

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=128)
test_loader = DataLoader(test_dataset, batch_size=128)

for batch in train_loader:
    batch.x = F.one_hot(batch.x.squeeze(-1), num_classes=28).to(torch.float)
