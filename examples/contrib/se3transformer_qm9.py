import os.path as osp
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

import torch_geometric.transforms as T
from torch_geometric.contrib.nn import SE3Transformer
from torch_geometric.contrib.utils import Fiber
from torch_geometric.data import Data
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

TASK_CHOICES = [
    "mu",
    "alpha",
    "homo",
    "lumo",
    "gap",
    "r2",
    "zpve",
    "U0",
    "U",
    "H",
    "G",
    "Cv",
    "U0_atom",
    "U_atom",
    "H_atom",
    "G_atom",
    "A",
    "B",
    "C",
]
data_path = osp.join(osp.dirname(osp.realpath(__file__)), "data", "QM9")


class ChooseTarget(T.BaseTransform):
    def __init__(self, target: int) -> None:
        self.target = target

    def __call__(self, data: Data) -> Data:
        data.y = data.y[:, self.target]
        return data


class AsFiber(T.BaseTransform):
    def __init__(self, node_fiber: Fiber, edge_fiber: Fiber) -> None:
        self.node_fiber = node_fiber
        self.edge_fiber = edge_fiber

    def __call__(self, data: Data) -> Data:
        data.node_feats = self.node_fiber.tensor_to_dict(data.x)
        data.edge_feats = self.edge_fiber.tensor_to_dict(data.edge_attr)
        return data


def train_step(epoch, model, train_loader, optimizer, device, loss_fn):
    model.train()
    loss_all = 0

    for data in tqdm(train_loader, unit="batch", desc=f"Epoch {epoch}"):
        data = data.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(data), data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    return loss_all / len(train_loader)


def train(
    model: nn.Module,
    loss_fn: nn.modules.loss._Loss,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    std,
    epochs: int = 100,
    lr=0.002,
):
    device = torch.cuda.current_device()
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.7, patience=5, min_lr=0.00001
    )
    for epoch in range(epochs):
        lr = scheduler.optimizer.param_groups[0]['lr']
        loss = train_step(epoch, model, train_loader, optimizer, device, loss_fn)
        print(f"Training Loss: {loss / len(train_dataloader)}")
        if epoch > 0 and epoch % 1 == 0:
            eval_accuracy = test(model, val_dataloader, std)
            print(f"Eval MAE: {eval_accuracy}")
            model.train()


def test(model: torch.nn.Module, dataloader: DataLoader, std):
    error = 0
    total = 0
    error_fn = torch.nn.L1Loss(reduction="sum")
    model.eval()
    for batch in tqdm(dataloader, unit="batch", desc=f"Evaluation"):
        batch = batch.cuda()
        preds = model(batch).detach()
        n = preds.shape[0]
        error += error_fn(preds, batch.y)
        total += n
    return (error / total) * std


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="SE(3)-Transformer")
    PARSER.add_argument(
        "--num_degrees",
        help="Number of degrees to use. Hidden features will have types [0, ..., num_degrees - 1]",
        type=int,
        default=4,
    )
    PARSER.add_argument(
        "--num_channels",
        help="Number of channels for the hidden features",
        type=int,
        default=32,
    )
    args = PARSER.parse_args()

    batch_size = 480
    num_workers = 4
    task = "homo"
    target = TASK_CHOICES.index(task)

    node_feature_dim = 6  # Keep only the first 6 features to match the original paper
    edge_feature_dim = 5  # The original 4 features, plus one for distance
    fiber_in = Fiber({0: node_feature_dim})
    fiber_edge = Fiber({0: edge_feature_dim})

    transform = T.Compose(
        [
            ChooseTarget(target),
            T.Distance(norm=False),
            AsFiber(fiber_in, fiber_edge),
            T.Cartesian(norm=False),
        ]
    )

    dataset = QM9(data_path, transform=transform)

    mean = dataset.data.y.mean(dim=0, keepdim=True)
    std = dataset.data.y.std(dim=0, keepdim=True)
    dataset.data.y = (dataset.data.y - mean) / std
    mean, std = mean[:, target].item(), std[:, target].item()

    test_dataset = dataset[:10000]
    val_dataset = dataset[10000:20000]
    train_dataset = dataset[20000:]
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    model = SE3Transformer(
        fiber_in=fiber_in,
        fiber_hidden=Fiber.create(args.num_degrees, args.num_channels),
        fiber_out=Fiber({0: args.num_degrees * args.num_channels}),
        fiber_edge=fiber_edge,
        output_dim=1,
        pooling="max",
        num_layers=7,
        num_heads=8,
        channels_div=2,
        return_type=0
        # **vars(args),
    )
    loss_fn = nn.L1Loss()

    print("====== SE(3)-Transformer ======")
    print("|      Training procedure     |")
    print("===============================")

    train(model, loss_fn, train_loader, val_loader, std)
    test_accuracy = test(model, test_loader)
    print(f"Test Accuracy: {test_accuracy}")
