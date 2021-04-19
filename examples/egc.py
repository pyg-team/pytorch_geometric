import os.path as osp
import argparse

import torch
import torch.nn.functional as F
from torch.nn import ModuleList, BatchNorm1d
from torch.nn import Sequential, ReLU, Linear
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import global_mean_pool, EGConv
from torch_geometric.data import DataLoader

from ogb.graphproppred import PygGraphPropPredDataset
from ogb.graphproppred.evaluate import Evaluator
from ogb.graphproppred.mol_encoder import AtomEncoder

parser = argparse.ArgumentParser()
parser.add_argument('--hidden', type=int, default=236)
parser.add_argument('--layers', type=int, default=4)
parser.add_argument('--heads', type=int, default=4)
parser.add_argument('--bases', type=int, default=4)
parser.add_argument('--use_multi_aggrs',
                    action="store_true",
                    help="Switch between EGC-S and EGC-M")
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--lr', type=float, default=1e-4)
args = parser.parse_args()

path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data",
                "ogbg-molhiv")

dataset = PygGraphPropPredDataset(name=f"ogbg-molhiv", root=path)
split_idx = dataset.get_idx_split()
loaders = dict()
for split in ["train", "valid", "test"]:
    loaders[split] = DataLoader(
        dataset[split_idx[split]],
        batch_size=args.batch_size,
        shuffle=(split == "train"),
        num_workers=args.num_workers,
    )


class Net(torch.nn.Module):
    def __init__(self, ):
        super().__init__()
        hidden = args.hidden
        aggs = ["sum", "mean", "max"] if args.use_multi_aggrs else ["symnorm"]
        self.embedding = AtomEncoder(hidden)
        self.graph_layers = ModuleList()

        for _ in range(args.layers):
            self.graph_layers.append(
                ModuleList([
                    EGConv(
                        hidden,
                        hidden,
                        aggrs=aggs,
                        num_heads=args.heads,
                        num_bases=args.bases,
                    ),
                    BatchNorm1d(hidden),
                    ReLU(),
                ]))

        self.pool = global_mean_pool
        self.mlp = Sequential(
            Linear(hidden, hidden // 2, bias=False),
            BatchNorm1d(hidden // 2),
            ReLU(),
            Linear(hidden // 2, hidden // 4, bias=False),
            BatchNorm1d(hidden // 4),
            ReLU(),
            Linear(hidden // 4, 1),
        )

    def forward(self, batch):
        x, edge_index = batch.x, batch.edge_index
        x = self.embedding(x.squeeze())

        for gcn, bn, act in self.graph_layers:
            identity = x
            x = gcn(x=x, edge_index=edge_index)
            x = bn(x)
            x = act(x)
            x += identity

        x = self.pool(x, batch.batch)
        return self.mlp(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = ReduceLROnPlateau(optimizer,
                              mode="max",
                              factor=0.5,
                              patience=20,
                              min_lr=1e-5)
evaluator = Evaluator("ogbg-molhiv")


def train():
    model.train()
    num_batches = 0
    loss_total = 0.0

    for batch in loaders["train"]:
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch)
        # nan targets (unlabeled) should be ignored when computing training loss
        is_labeled = batch.y == batch.y
        loss = F.binary_cross_entropy_with_logits(
            out.to(torch.float32)[is_labeled],
            batch.y.to(torch.float32)[is_labeled])
        loss.backward()
        optimizer.step()

        loss_total += loss.item()
        num_batches += 1

    return loss_total / num_batches


@torch.no_grad()
def evaluate(split):
    model.eval()

    y_true = []
    y_pred = []

    for batch in loaders[split]:
        batch = batch.to(device)
        pred = model(batch)
        y_true.append(batch.y.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()
    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)["rocauc"]


for epoch in range(1, args.epochs + 1):
    loss = train()
    val_metric = evaluate("valid")
    test_metric = evaluate("test")
    scheduler.step(val_metric)
    print(f"Epoch: {epoch:02d}, Loss: {loss:.4f}, Val: {val_metric:.4f}, "
          f"Test: {test_metric:.4f}")
