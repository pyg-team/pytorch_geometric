import os.path as osp

import torch
from torch_geometric.datasets import FAUST
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'FAUST')
aug_slide_verts = 0.0
aug_scale_verts = True
aug_flip_edges = 0.5

pre_transform_train = T.Compose(
    [T.MeshCNNPrepare(aug_slide_verts, aug_scale_verts, aug_flip_edges)])
pre_transform_test = T.Compose([T.MeshCNNPrepare()])
train_dataset = FAUST(path, True, T.Cartesian(), pre_transform_train)
test_dataset = FAUST(path, False, T.Cartesian(), pre_transform_test)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1)
d = train_dataset[0]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
target = torch.arange(d.num_nodes, dtype=torch.long, device=device)

for data in train_loader:
    print('This is a test for FAUST dataset with MeshCNN structure')
