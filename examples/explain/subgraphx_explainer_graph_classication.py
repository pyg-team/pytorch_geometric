import random

import networkx as nx
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.nn import Linear
from tqdm import tqdm

import torch_geometric.nn as nn
from torch_geometric import seed_everything
from torch_geometric.data import Data
from torch_geometric.datasets import BA2MotifDataset
from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import SubgraphXExplainer
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx


class GCN(torch.nn.Module):
    def __init__(
        self,
        num_node_features: int,
        num_classes: int,
        hidden_channels=20,
        batch_norm=True,
    ):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.batch_norm = batch_norm
        self.conv1 = nn.conv.GCNConv(num_node_features, hidden_channels)
        self.conv2 = nn.conv.GCNConv(hidden_channels, hidden_channels)
        self.conv3 = nn.conv.GCNConv(hidden_channels, hidden_channels)

        if self.batch_norm:
            self.bn1 = nn.norm.GraphNorm(hidden_channels)
            self.bn2 = nn.norm.GraphNorm(hidden_channels)

        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch=None):
        # for a single point, batch might not be passed
        if batch is None:
            batch = torch.zeros(x.size()[0], dtype=torch.long).to(x.device)

        if self.batch_norm:
            x = self.bn1(self.conv1(x, edge_index).relu(), batch)
            x = self.bn2(self.conv2(x, edge_index).relu(), batch)
        else:
            x = self.conv1(x, edge_index).relu()
            x = self.conv2(x, edge_index).relu()

        x = self.conv3(x, edge_index)
        x = nn.pool.global_max_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


# Helper: Train function
def train(model, dl, optimizer, loss_fn):
    """Trains the model

    Args:
    model: torch.nn.Module: Pytorch Network
    dl: Pytorch Dataloader
    optimizer: torch.optim Optimizer
    loss_fn: loss function
    """
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(dl):
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = loss_fn(out, batch.y)

        # optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update total_loss
        total_loss = loss.detach().cpu()

    return total_loss


# Helper: Get Accuracy function
@torch.no_grad()
def get_acc(model, dl):
    """Computes Accuracy given model and dataloader"""
    model.eval()
    preds, gts = [], []
    for batch_idx, batch in enumerate(dl):
        pred = model(batch.x, batch.edge_index, batch.batch)

        preds.append(pred.argmax(1).detach().cpu())
        gts.append(batch.y.detach().cpu())

    # stack and compute acc
    preds_ = torch.cat([o for o in preds], dim=0)
    gts_ = torch.cat([o for o in gts], dim=0)
    return torch.mean((preds_ == gts_) * 1.0).item()


def main():
    # Seed everything
    seed_everything(42)

    # initialize dataset
    dataset = BA2MotifDataset(root="data/")

    # create train and test dl
    idx = torch.arange(len(dataset))
    train_idx, test_idx = train_test_split(idx, train_size=0.8, stratify=dataset.data.y)

    train_dataset = dataset[train_idx]
    test_dataset = dataset[test_idx]

    train_dl = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dl = DataLoader(test_dataset, batch_size=64)

    # Train the GCN for graph classification
    model = GCN(
        num_node_features=dataset.num_node_features,
        num_classes=dataset.num_classes,
        batch_norm=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()

    epochs = 200
    pbar = tqdm(range(1, epochs + 1))
    for epoch in pbar:
        # train the model
        loss = train(model, train_dl, optimizer, loss_fn)
        if epoch == 1 or epoch % 10 == 0:
            train_acc, test_acc = (get_acc(model, train_dl), get_acc(model, test_dl))
            pbar.set_description(
                f"Loss: {loss:.4f}, Train: {train_acc:.4f}," f"Test: {test_acc:.4f}"
            )

    pbar.close()
    # set model to eval mode
    model.eval()

    # Initialize Explainer
    # As we have 3 GCNConv Layers
    num_layers = 3
    explainer = Explainer(
        model=model,
        algorithm=SubgraphXExplainer(
            num_classes=dataset.num_classes,
            device=torch.device("cpu"),
            num_hops=num_layers,
            max_nodes=5,
            rollout=2,
            reward_method="mc_l_shapley",
        ),
        explanation_type="model",
        node_mask_type="object",
        edge_mask_type="object",
        model_config=dict(
            mode="multiclass_classification",
            task_level="graph",
            return_type="raw",
        ),
    )

    # let's get an explanation for node_idx (seed=42)
    node_idx = random.choice(idx).item()
    explanation = explainer(dataset[node_idx].x, dataset[node_idx].edge_index)

    # we can plot the explanation using networkx
    G = to_networkx(
        Data(x=dataset[node_idx].x, edge_index=dataset[node_idx].edge_index)
    )
    pos = nx.spring_layout(G)
    G = G.to_undirected()
    nx.draw_networkx(G, pos=pos)
    nx.draw_networkx(
        G.subgraph(explanation.masked_node_list), pos=pos, node_color="red"
    )


if __name__ == "__main__":
    main()
