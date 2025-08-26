import os
from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
    BooleanOptionalAction,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.graphproppred import Evaluator, PygGraphPropPredDataset
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch import Tensor
from tqdm import tqdm

from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.models import GCN, TokenGT
from torch_geometric.transforms import AddOrthonormalNodeIdentifiers

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--use_gcn", type=bool, default=False,
                    action=BooleanOptionalAction)
parser.add_argument("--lr", type=float, default=0.00001)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--n_epochs", type=int, default=50)
args = parser.parse_args()

if args.use_gcn is True:
    print("Model choice: GCN")
else:
    print("Model choice: TokenGT")


class TokenGTGraphBinClassifier(nn.Module):
    def __init__(
        self,
        dim_node,
        d_p,
        d,
        num_heads,
        num_encoder_layers,
        dim_feedforward,
        is_laplacian_node_ids,
        dropout,
        device,
    ):
        super().__init__()
        self._atom_encoder = AtomEncoder(dim_node).to(device)
        self._bond_encoder = BondEncoder(dim_node).to(device)

        self._token_gt = TokenGT(
            dim_node=dim_node,
            dim_edge=dim_node,
            d_p=d_p,
            d=d,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            is_laplacian_node_ids=is_laplacian_node_ids,
            include_graph_token=True,
            dropout=dropout,
            device=device,
        )
        self._lm = nn.Linear(d, 1, device=device)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        ptr: Tensor,
        batch: Tensor,
        node_ids: Tensor,
    ):
        x, edge_attr = self._atom_encoder(x), self._bond_encoder(edge_attr)
        _, graph_emb = self._token_gt(x, edge_index, edge_attr, ptr, batch,
                                      node_ids)
        logit = self._lm(graph_emb)
        return logit


class GCNGraphBinClassifier(torch.nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_layers,
        dropout,
        device,
    ):
        super().__init__()

        self._atom_encoder = AtomEncoder(hidden_dim).to(device)
        self._bond_encoder = BondEncoder(hidden_dim).to(device)
        self._gcn = GCN(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            num_layers=num_layers,
            out_channels=hidden_dim,
            dropout=dropout,
        ).to(device)
        self._lm = torch.nn.Linear(hidden_dim, 1, device=device)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch: Tensor,
        **kwargs,
    ):
        x, edge_attr = self._atom_encoder(x), self._bond_encoder(edge_attr)
        x_emb = self._gcn(x, edge_index, edge_attr=edge_attr)
        graph_emb = global_mean_pool(x_emb, batch)
        logit = self._lm(graph_emb)
        return logit


D_P = 64
transform = AddOrthonormalNodeIdentifiers(D_P, True)
# note: use pre_transform (avoid unnecessary duplicate eigenvector calculation)
dataset = PygGraphPropPredDataset(name="ogbg-molhiv", root="./data",
                                  pre_transform=transform)
idx_split = dataset.get_idx_split()
train_dataset = dataset[idx_split["train"]]
valid_dataset = dataset[idx_split["valid"]]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                          num_workers=8)
eval_train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False,
                               num_workers=8)
valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=False,
                          num_workers=8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, loader, optimizer, device):
    model.train()
    for batch in tqdm(loader, desc="Iteration"):
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            ptr=batch.ptr,
            batch=batch.batch,
            node_ids=batch.node_ids,
        )
        loss = F.binary_cross_entropy_with_logits(out.squeeze(1),
                                                  batch.y.squeeze(1).float())
        loss.backward()
        optimizer.step()


def get_loss_and_auc(model, loader, device, evaluator):
    model.eval()
    total_loss = 0.0
    y_true, y_pred = [], []
    for batch in tqdm(loader, desc="Iteration"):
        batch = batch.to(device)
        with torch.no_grad():
            out = model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                ptr=batch.ptr,
                batch=batch.batch,
                node_ids=batch.node_ids,
            )
            loss = F.binary_cross_entropy_with_logits(
                out.squeeze(1),
                batch.y.squeeze(1).float())
            total_loss += float(loss)

            y_true.append(batch.y.view(out.shape).detach())
            y_pred.append(out.detach())

    avg_loss = total_loss / len(loader)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_pred = torch.cat(y_pred, dim=0).cpu().numpy()
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    auc = evaluator.eval(input_dict)["rocauc"]
    return avg_loss, auc


if args.use_gcn:
    model = GCNGraphBinClassifier(
        hidden_dim=256,
        num_layers=5,
        dropout=args.dropout,
        device=device,
    )
else:
    model = TokenGTGraphBinClassifier(
        dim_node=256,
        d_p=D_P,
        d=256,
        num_heads=8,
        num_encoder_layers=4,
        dim_feedforward=512,
        is_laplacian_node_ids=True,
        dropout=args.dropout,
        device=device,
    )

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
evaluator = Evaluator("ogbg-molhiv")
if not os.path.exists("./models"):
    os.mkdir("./models")

train_loss, train_auc = get_loss_and_auc(model, eval_train_loader, device,
                                         evaluator)
valid_loss, valid_auc = get_loss_and_auc(model, valid_loader, device,
                                         evaluator)
valid_aucs = {}
print(f"Epoch 0: train loss: {train_loss} valid loss: {valid_loss}")
print(f"Epoch 0: train roc-auc: {train_auc} valid roc-auc: {valid_auc}")
for epoch in range(1, args.n_epochs + 1):
    train(model, train_loader, optimizer, device)
    train_loss, train_auc = get_loss_and_auc(model, eval_train_loader, device,
                                             evaluator)
    valid_loss, valid_auc = get_loss_and_auc(model, valid_loader, device,
                                             evaluator)
    print(f"Epoch {epoch}: train loss: {train_loss} valid loss: {valid_loss}")
    print(
        f"Epoch {epoch}: train roc-auc: {train_auc} valid roc-auc: {valid_auc}"
    )

    # save model and record valid auc
    torch.save(model.state_dict(), f'./models/{epoch}.pt')
    valid_aucs[epoch] = valid_auc

max_valid_auc_k = max(valid_aucs, key=lambda x: valid_aucs[x])
model.load_state_dict(torch.load(f"./models/{max_valid_auc_k}.pt"))

_, best_valid_auc = get_loss_and_auc(model, valid_loader, device, evaluator)
print(f"Best valid roc-auc: {best_valid_auc}")
