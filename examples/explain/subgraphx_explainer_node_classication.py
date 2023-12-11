import random

import torch
import torch.nn.functional as F
from dig.xgraph.method.subgraphx import PlotUtils
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch_geometric.transforms as T
from torch_geometric import seed_everything
from torch_geometric.data import Data
from torch_geometric.datasets import ExplainerDataset
from torch_geometric.datasets.graph_generator import BAGraph
from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm.subgraphX_explainer import (
    MCTS,
    SubgraphXExplainer,
)
from torch_geometric.nn import GCN
from torch_geometric.utils import to_networkx


def train(model, data, optimizer, train_idx):
    """Trains the model"""
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[train_idx], data.y[train_idx])
    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def get_acc(model, data, train_idx, test_idx):
    """Returns train and test accuracy"""
    model.eval()
    pred = model(data.x, data.edge_index).argmax(dim=-1)

    train_correct = int((pred[train_idx] == data.y[train_idx]).sum())
    train_acc = train_correct / train_idx.size(0)

    test_correct = int((pred[test_idx] == data.y[test_idx]).sum())
    test_acc = test_correct / test_idx.size(0)

    return train_acc, test_acc


def main():
    # set seed
    seed_everything(42)

    # get data
    dataset = ExplainerDataset(
        graph_generator=BAGraph(num_nodes=50, num_edges=15),
        motif_generator="house",
        num_motifs=20,
        transform=T.Constant(),
    )
    data = dataset[0]

    # set number of GCN Convolutions (i.e. num_hops)
    num_layers = 3
    idx = torch.arange(data.num_nodes)
    train_idx, test_idx = train_test_split(idx, train_size=0.75, stratify=data.y)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    model = GCN(
        data.num_node_features,
        hidden_channels=30,
        num_layers=num_layers,
        out_channels=dataset.num_classes,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)

    # train our model
    pbar = tqdm(range(1, 2001))
    for epoch in pbar:
        loss = train(model, data, optimizer, train_idx)
        if epoch == 1 or epoch % 200 == 0:
            train_acc, test_acc = get_acc(model, data, train_idx, test_idx)
            pbar.set_description(
                f"Loss: {loss:.4f}, Train: {train_acc:.4f}, " f"Test: {test_acc:.4f}"
            )
    pbar.close()
    model.eval()

    # get index of non-zero label node and where our prediction is correct
    logits = model(data.x, data.edge_index)
    prediction = logits.argmax(dim=-1)

    # get list of correct predictions and random.choice() out of it
    indices = ((prediction == data.y) & (prediction != 0)).nonzero()
    node_idx = random.choice(indices.squeeze().tolist())

    # Initialize our explainer
    explainer = Explainer(
        model=model,
        algorithm=SubgraphXExplainer(
            num_classes=4,
            device=device,
            num_hops=num_layers,
            max_nodes=5,
            rollout=2,
            reward_method="l_shapley",
        ),
        explanation_type="model",
        node_mask_type="object",
        edge_mask_type="object",
        model_config=dict(
            mode="multiclass_classification",
            task_level="node",
            return_type="raw",
        ),
    )
    # get our explanation
    explanation = explainer(data.x, data.edge_index, index=node_idx)

    # Get subgraph for our node_idx based on num_layers (or num_hops)
    # Convert subgraph to networkx object for visualization
    subgraph_x, subgraph_edge_index, subset, _, _ = MCTS.__subgraph__(
        node_idx, data.x, data.edge_index, num_hops=num_layers
    )
    subgraph_y = data.y[subset].to("cpu")
    G = to_networkx(Data(x=subgraph_x, edge_index=subgraph_edge_index))
    new_node_idx = torch.where(subset == node_idx)[0].item()

    # plot using PlotUtils
    plotutils = PlotUtils(dataset_name="ba_shapes")
    plotutils.plot(
        G,
        nodelist=explanation.masked_node_list,
        figname=None,
        y=subgraph_y,
        node_idx=new_node_idx,
    )
