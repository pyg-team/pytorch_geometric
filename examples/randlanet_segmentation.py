import os.path as osp

import torch
import torch.nn.functional as F
from torch_scatter import scatter
from torchmetrics.functional import jaccard_index
from tqdm import tqdm

import torch_geometric.transforms as T
from torch_geometric.datasets import ShapeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, knn_interpolate
from randlanet_classification import DilatedResidualBlock


category = "Airplane"  # Pass in `None` to train on all categories.
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
    def __init__(self, k, nn):
        super().__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip


class Net(torch.nn.Module):
    def __init__(
        self, num_features, num_classes, decimation: int = 4, num_neighboors: int = 16
    ):
        super().__init__()
        self.lfa1_module = DilatedResidualBlock(
            decimation, num_neighboors, num_features, 32
        )
        self.lfa2_module = DilatedResidualBlock(decimation, num_neighboors, 32, 128)
        self.lfa3_module = DilatedResidualBlock(decimation, num_neighboors, 128, 256)
        self.lfa4_module = DilatedResidualBlock(decimation, num_neighboors, 256, 512)
        self.mlp1 = MLP([512, 512, 512])
        self.fp4_module = FPModule(1, MLP([512 + 256, 256]))
        self.fp3_module = FPModule(1, MLP([256 + 128, 128]))
        self.fp2_module = FPModule(1, MLP([128 + 32, 32]))
        self.fp1_module = FPModule(1, MLP([32 + num_features, 8]))

        self.mlp2 = MLP([8, 64, 32], dropout=0.5)
        self.lin = torch.nn.Linear(32, num_classes)

    def forward(self, data):
        in_0 = (data.x, data.pos, data.batch)

        lfa1_out = self.lfa1_module(*in_0)
        lfa2_out = self.lfa2_module(*lfa1_out)
        lfa3_out = self.lfa3_module(*lfa2_out)
        lfa4_out = self.lfa4_module(*lfa3_out)

        mlp_out = (self.mlp1(lfa4_out[0]), lfa4_out[1], lfa4_out[2])

        fp4_out = self.fp4_module(*mlp_out, *lfa3_out)
        fp3_out = self.fp3_module(*fp4_out, *lfa2_out)
        fp2_out = self.fp2_module(*fp3_out, *lfa1_out)
        x, _, _ = self.fp1_module(*fp2_out, *in_0)

        x = self.mlp2(x)

        return self.lin(x).log_softmax(dim=-1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net(3, train_dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


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
