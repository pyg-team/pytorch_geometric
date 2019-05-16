import torch
import numpy as np

import torch_geometric.transforms as GT
from torch_geometric.data import DataLoader

from shapenet2 import ShapeNet2
from pointnet2_part_seg import PointNet2PartSegmentNet
from solver import train_epoch, test_epoch, eval_mIOU

# Configuration
shapenet_dataset = '../../data/shapenet'
category = 'Airplane'
# npoints = 0  # Do not sample when npoints <= 0
npoints = 2500  # Sample to fixed number of points when npoints > 0
batch_size = 8
num_workers = 6
nepochs = 25

manual_seed = 123
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)
torch.cuda.manual_seed(manual_seed)

# Transform
rot_max_angle = 15
trans_max_distance = 0.01

RotTransform = GT.Compose([
    GT.RandomRotate(rot_max_angle, 0),
    GT.RandomRotate(rot_max_angle, 1),
    GT.RandomRotate(rot_max_angle, 2)
])
TransTransform = GT.RandomTranslate(trans_max_distance)

train_transform = GT.Compose(
    [GT.NormalizeScale(), RotTransform, TransTransform])
test_transform = GT.Compose([
    GT.NormalizeScale(),
])

# Dataset
train_dataset = ShapeNet2(
    shapenet_dataset,
    npoints=npoints,
    category=category,
    train=True,
    transform=train_transform)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers)
num_classes = train_dataset.num_classes

test_dataset = ShapeNet2(
    shapenet_dataset,
    npoints=npoints,
    category=category,
    train=False,
    transform=test_transform)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers)

# Model, criterion and optimizer
device = torch.device('cuda') if torch.cuda.is_available() else torch.device(
    'cpu')
dtype = torch.float

model = PointNet2PartSegmentNet(num_classes).to(device, dtype)
criterion = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters())

# Train
for epoch in range(0, nepochs):
    print('Epoch {}/{}'.format(epoch + 1, nepochs))
    train_epoch(model, criterion, optimizer, train_dataloader, device)
    test_epoch(model, test_dataloader, device)

# mIOU
mIOU = eval_mIOU(model, test_dataloader, device, num_classes)
print('mIOU for category {}: {}'.format(category, mIOU))
