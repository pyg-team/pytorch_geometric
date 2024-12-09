import os.path as osp
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.nn import TokenGT
from torch_geometric.transforms import AddOrthonormalNodeIdentifiers


class TokenGTGraphRegression(nn.Module):
    def __init__(
        self,
        dim_node,
        d_p,
        d,
        num_heads,
        num_encoder_layers,
        dim_feedforward,
        is_laplacian_node_ids,
        dim_edge,
        dropout,
        device,
    ):
        super().__init__()
        self._token_gt = TokenGT(
            dim_node=dim_node,
            d_p=d_p,
            d=d,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dim_edge=dim_edge,
            is_laplacian_node_ids=is_laplacian_node_ids,
            include_graph_token=True,
            dropout=dropout,
            device=device,
        )
        self.lm = nn.Linear(d, 1, device=device)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor],
        ptr: Tensor,
        batch: Tensor,
        node_ids: Tensor,
    ):
        _, graph_emb = self._token_gt(x, edge_index, edge_attr, ptr, batch,
                                      node_ids)
        return self.lm(graph_emb)


def train(model, loader, criterion, optimizer):
    model.train()
    for batch in loader:
        optimizer.zero_grad()
        out = model(
            batch.x.float(),
            batch.edge_index,
            batch.edge_attr.unsqueeze(1).float(),
            batch.ptr,
            batch.batch,
            batch.node_ids,
        )
        loss = criterion(out, batch.y.unsqueeze(1))
        loss.backward()
        optimizer.step()


def get_test_loss(model, loader) -> float:
    criterion = nn.L1Loss(reduction="sum")
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            out = model(
                batch.x.float(),
                batch.edge_index,
                batch.edge_attr.unsqueeze(1).float(),
                batch.ptr,
                batch.batch,
                batch.node_ids,
            )
            loss = criterion(out, batch.y.unsqueeze(1)).item()
            total_loss += loss
    return total_loss / len(loader.dataset)


D_P = 37

transform = AddOrthonormalNodeIdentifiers(D_P, True)
path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data", "ZINC-lap")
# note: use pre_transform (avoid unnecessary duplicate eigenvector calculation)
train_dataset = ZINC(path, subset=True, split="train", pre_transform=transform)
test_dataset = ZINC(path, subset=True, split="test", pre_transform=transform)

if torch.cuda.is_available():
    train_dataset.cuda()
    test_dataset.cuda()

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TokenGTGraphRegression(
    dim_node=train_dataset.num_node_features,
    d_p=D_P,
    d=32,
    num_heads=4,
    num_encoder_layers=2,
    dim_feedforward=64,
    is_laplacian_node_ids=True,
    dim_edge=train_dataset.num_edge_features,
    dropout=0.1,
    device=device,
)

criterion = nn.L1Loss(reduction="mean")
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001)

train_loss = round(get_test_loss(model, train_loader), 5)
test_loss = round(get_test_loss(model, test_loader), 5)
print(f"Epoch 0: train_loss={train_loss} test_loss={test_loss}")

for i in range(100):
    train(model, train_loader, criterion, optimizer)
    train_loss = round(get_test_loss(model, train_loader), 5)
    test_loss = round(get_test_loss(model, test_loader), 5)
    print(f"Epoch {i+1}: train_loss={train_loss} test_loss={test_loss}")
