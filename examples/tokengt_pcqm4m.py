import argparse
import os.path as osp

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

from torch_geometric.datasets import PCQM4Mv2
from torch_geometric.loader import DataLoader
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import TokenGT
from torch_geometric.transforms import AddTokenGTLaplacianNodeIds
from torch_geometric.data import Data
from torch_geometric.transforms.base_transform import BaseTransform
from torch_geometric.transforms import Compose
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--d_p", type=int, default=64)
parser.add_argument(
    "--node_id_mode", type=str, default="laplacian", choices=["orf", "laplacian"]
)
parser.add_argument("--embedding_dim", type=int, default=768)
parser.add_argument("--num_attention_heads", type=int, default=32)
parser.add_argument("--num_encoder_layers", type=int, default=12)
parser.add_argument("--ffn_embedding_dim", type=int, default=768)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--lr", type=float, default=0.0002)
parser.add_argument("--weight_decay", type=float, default=0.1)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=400)
parser.add_argument("--warmup_steps", type=int, default=2000)
parser.add_argument("--lap_node_id_eig_dropout", type=float, default=0.2)
parser.add_argument("--lap_node_id_sign_flip", action="store_true", default=True)
parser.add_argument("--encoder_normalize_before", action="store_true", default=True)
parser.add_argument("--wandb", action="store_true")
parser.add_argument("--fp16", action="store_true")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = GradScaler() if args.fp16 and device.type == "cuda" else None


class LinearSchedulerWithWarmup:
    def __init__(
        self,
        optimizer,
        num_warmup_steps,
        num_training_steps,
        last_epoch=-1,
        min_lr_ratio=0.1,
    ):
        self.optimizer = optimizer
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.last_epoch = last_epoch
        self.base_lr = optimizer.param_groups[0]["lr"]
        self.min_lr_ratio = min_lr_ratio

    def step(self):
        self.last_epoch += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_lr(self):
        if self.last_epoch < self.num_warmup_steps:
            # Linear warmup
            return self.base_lr * self.last_epoch / self.num_warmup_steps
        else:
            # Linear decay
            progress = (self.last_epoch - self.num_warmup_steps) / (
                self.num_training_steps - self.num_warmup_steps
            )
            return self.base_lr * (
                self.min_lr_ratio + (1 - self.min_lr_ratio) * (1 - progress)
            )


def convert_to_single_emb(x, offset: int = 512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = torch.arange(
        0, feature_num * offset, offset, dtype=torch.long, device=x.device
    )
    x = x + feature_offset
    return x


class ConvertToSingleEmbTransform(BaseTransform):
    def __init__(self, offset: int = 512):
        self.offset = offset

    def forward(self, data: Data) -> Data:
        data.x = convert_to_single_emb(data.x, self.offset)
        if data.edge_attr is not None:
            data.edge_attr = convert_to_single_emb(data.edge_attr, self.offset)
        return data


class TokenGTGraphRegression(nn.Module):
    def __init__(
        self,
        num_atoms,
        num_edges,
        d_p,
        embedding_dim,
        num_attention_heads,
        num_encoder_layers,
        ffn_embedding_dim,
        dropout,
        lap_node_id_eig_dropout,
        lap_node_id_sign_flip,
        encoder_normalize_before,
    ):
        super().__init__()
        self._token_gt = TokenGT(
            num_atoms=num_atoms,
            num_edges=num_edges,
            node_id_mode=args.node_id_mode,
            d_p=d_p,
            num_encoder_layers=num_encoder_layers,
            embedding_dim=embedding_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            lap_node_id_eig_dropout=lap_node_id_eig_dropout,
            lap_node_id_sign_flip=lap_node_id_sign_flip,
            encoder_normalize_before=encoder_normalize_before,
        )
        self.lm = nn.Linear(embedding_dim, 1)

    def forward(self, x, edge_index, edge_attr, ptr, batch, node_ids):
        _, _, graph_emb = self._token_gt(x, edge_index, edge_attr, ptr, batch, node_ids)
        return self.lm(graph_emb)


def train(scheduler):
    model.train()
    total_loss = 0.0
    for batch in tqdm(train_loader, desc="Training"):
        batch = batch.to(device)
        optimizer.zero_grad()

        if args.fp16 and scaler is not None:
            with autocast("cuda"):
                out = model(
                    batch.x,
                    batch.edge_index,
                    batch.edge_attr,
                    batch.ptr,
                    batch.batch,
                    batch.node_ids if hasattr(batch, "node_ids") else None,
                )
                loss = criterion(out, batch.y.unsqueeze(1))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.ptr,
                batch.batch,
                batch.node_ids if hasattr(batch, "node_ids") else None,
            )
            loss = criterion(out, batch.y.unsqueeze(1))
            loss.backward()
            optimizer.step()

        scheduler.step()
        total_loss += float(loss.detach())
    return total_loss / len(train_loader)


@torch.no_grad()
def test():
    model.eval()
    total_loss = 0.0
    for batch in test_loader:
        batch = batch.to(device)

        if args.fp16 and scaler is not None:
            with autocast("cuda"):
                out = model(
                    batch.x,
                    batch.edge_index,
                    batch.edge_attr,
                    batch.ptr,
                    batch.batch,
                    batch.node_ids if hasattr(batch, "node_ids") else None,
                )
                loss = criterion(out, batch.y.unsqueeze(1))
        else:
            out = model(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.ptr,
                batch.batch,
                batch.node_ids if hasattr(batch, "node_ids") else None,
            )
            loss = criterion(out, batch.y.unsqueeze(1))

        total_loss += float(loss.detach())
    return total_loss / len(test_loader)


init_wandb(
    name=f"TokenGT-PCQM4M",
    d_p=args.d_p,
    embedding_dim=args.embedding_dim,
    num_attention_heads=args.num_attention_heads,
    num_encoder_layers=args.num_encoder_layers,
    ffn_embedding_dim=args.ffn_embedding_dim,
    dropout=args.dropout,
    lr=args.lr,
    epochs=args.epochs,
    batch_size=args.batch_size,
    fp16=args.fp16,
    weight_decay=args.weight_decay,
    device=device,
)

transforms = [ConvertToSingleEmbTransform(offset=512)]
if args.node_id_mode == "laplacian":
    transforms.append(AddTokenGTLaplacianNodeIds(args.d_p))
transform = Compose(transforms)

path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data", "PCQM4M")
train_dataset = PCQM4Mv2(path, split="train", transform=transform)
test_dataset = PCQM4Mv2(path, split="val", transform=transform)

train_loader = DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4)

model = TokenGTGraphRegression(
    num_atoms=512 * 9,
    num_edges=512 * 3,
    d_p=args.d_p,
    embedding_dim=args.embedding_dim,
    num_attention_heads=args.num_attention_heads,
    num_encoder_layers=args.num_encoder_layers,
    ffn_embedding_dim=args.ffn_embedding_dim,
    dropout=args.dropout,
    lap_node_id_eig_dropout=0.2,
    lap_node_id_sign_flip=True,
    encoder_normalize_before=True,
).to(device)

print("--- Config ---")
for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")

criterion = nn.L1Loss(reduction="mean")
optimizer = torch.optim.AdamW(
    model.parameters(), lr=args.lr, weight_decay=args.weight_decay
)

total_training_steps = len(train_loader) * args.epochs
scheduler = LinearSchedulerWithWarmup(
    optimizer,
    num_warmup_steps=args.warmup_steps,
    num_training_steps=total_training_steps,
)

best_test_loss = float("inf")
for epoch in range(1, args.epochs + 1):
    train_loss = train(scheduler)
    test_loss = test()
    if test_loss < best_test_loss:
        best_test_loss = test_loss
    current_lr = optimizer.param_groups[0]["lr"]
    log(Epoch=epoch, Loss=train_loss, Train=train_loss, Test=test_loss, LR=current_lr)
    print(
        f"Epoch {epoch:03d}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, LR: {current_lr:.6f}"
    )

print(f"Best test loss: {best_test_loss:.4f}")
