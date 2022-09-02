import os.path as osp
from typing import Tuple

import torch
from torch import Tensor
from torch.nn import Sequential, Linear
import torch.nn.functional as F
from randlanet_classification import DilatedResidualBlock, SharedMLP, decimate
from torch_scatter import scatter
from torchmetrics.functional import jaccard_index
from tqdm import tqdm

import torch_geometric.transforms as T
from torch_geometric.datasets import ShapeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import knn_interpolate


category = "Airplane"  # Pass in `None` to train on all categories.
category_num_classes = 4  # 4 for Airplane - see ShapeNet for details
path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data", "ShapeNet")
transform = T.Compose(
    [
        T.RandomJitter(0.01),
        T.RandomRotate(15, axis=0),
        T.RandomRotate(15, axis=1),
        T.RandomRotate(15, axis=2),
    ]
)
pre_transform = T.NormalizeScale()
train_dataset = ShapeNet(
    path, category, split="trainval", transform=transform, pre_transform=pre_transform
)
test_dataset = ShapeNet(path, category, split="test", pre_transform=pre_transform)
train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True, num_workers=6)
test_loader = DataLoader(test_dataset, batch_size=12, shuffle=False, num_workers=6)


class FPModule(torch.nn.Module):
    """Upsampling with a skip connection."""

    def __init__(self, k, nn):
        super().__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip


class Net(torch.nn.Module):
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        decimation: int = 4,
        num_neighbors: int = 16,
        return_logits: bool = False,
    ):
        super().__init__()

        # An option to return logits instead of probas
        self.decimation = decimation
        self.return_logits = return_logits

        # Authors use 8, which might become a bottleneck if num_classes>8 or num_features>8,
        # and even for the final MLP.
        d_bottleneck = max(32, num_classes, num_features)

        self.fc0 = Linear(num_features, d_bottleneck)
        self.block1 = DilatedResidualBlock(num_neighbors, d_bottleneck, 32)
        self.block2 = DilatedResidualBlock(num_neighbors, 32, 128)
        self.block3 = DilatedResidualBlock(num_neighbors, 128, 256)
        self.block4 = DilatedResidualBlock(num_neighbors, 256, 512)
        self.mlp_summit = SharedMLP([512, 512])
        self.fp4 = FPModule(1, SharedMLP([512 + 256, 256]))
        self.fp3 = FPModule(1, SharedMLP([256 + 128, 128]))
        self.fp2 = FPModule(1, SharedMLP([128 + 32, 32]))
        self.fp1 = FPModule(1, SharedMLP([32 + 32, d_bottleneck]))
        self.mlp_end = Sequential(
            SharedMLP([d_bottleneck, 64, 32], dropout=[0.0, 0.5]),
            Linear(32, num_classes),
        )

    def forward(self, batch):
        ptr = batch.ptr.clone()
        in_0 = (self.fc0(batch.x), batch.pos, batch.batch)

        b1_out = self.block1(*in_0)
        b1_out_decimated, ptr = decimate(b1_out, ptr, self.decimation)

        b2_out = self.block2(*b1_out_decimated)
        b2_out_decimated, ptr = decimate(b2_out, ptr, self.decimation)

        b3_out = self.block3(*b2_out_decimated)
        b3_out_decimated, ptr = decimate(b3_out, ptr, self.decimation)

        b4_out = self.block4(*b3_out_decimated)
        b4_out_decimated, ptr = decimate(b4_out, ptr, self.decimation)

        mlp_out = (
            self.mlp_summit(b4_out_decimated[0]),
            b4_out_decimated[1],
            b4_out_decimated[2],
        )

        fp4_out = self.fp4(*mlp_out, *b3_out_decimated)
        fp3_out = self.fp3(*fp4_out, *b2_out_decimated)
        fp2_out = self.fp2(*fp3_out, *b1_out_decimated)
        fp1_out = self.fp1(*fp2_out, *b1_out)

        logits = self.mlp_end(fp1_out[0])

        if self.return_logits:
            return logits

        probas = logits.log_softmax(dim=-1)

        return probas


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net(3, category_num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train():
    model.train()

    total_loss = correct_nodes = total_nodes = 0
    for i, data in tqdm(enumerate(train_loader)):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct_nodes += out.argmax(dim=1).eq(data.y).sum().item()
        total_nodes += data.num_nodes

        if (i + 1) % 10 == 0:
            print(
                f"[{i+1}/{len(train_loader)}] Loss: {total_loss / 10:.4f} "
                f"Train Acc: {correct_nodes / total_nodes:.4f}"
            )
            total_loss = correct_nodes = total_nodes = 0


@torch.no_grad()
def test(loader):
    model.eval()

    ious, categories = [], []
    y_map = torch.empty(loader.dataset.num_classes, device=device).long()
    for data in loader:
        data = data.to(device)
        outs = model(data)

        sizes = (data.ptr[1:] - data.ptr[:-1]).tolist()
        for out, y, category in zip(
            outs.split(sizes), data.y.split(sizes), data.category.tolist()
        ):
            category = list(ShapeNet.seg_classes.keys())[category]
            part = ShapeNet.seg_classes[category]
            part = torch.tensor(part, device=device)

            y_map[part] = torch.arange(part.size(0), device=device)

            iou = jaccard_index(
                out[:, part].argmax(dim=-1),
                y_map[y],
                num_classes=part.size(0),
                absent_score=1.0,
            )
            ious.append(iou)

        categories.append(data.category)

    iou = torch.tensor(ious, device=device)
    category = torch.cat(categories, dim=0)

    mean_iou = scatter(iou, category, reduce="mean")  # Per-category IoU.
    return float(mean_iou.mean())  # Global IoU.


for epoch in range(1, 31):
    train()
    iou = test(test_loader)
    print(f"Epoch: {epoch:02d}, Test IoU: {iou:.4f}")
