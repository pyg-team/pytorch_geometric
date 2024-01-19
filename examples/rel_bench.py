import argparse
import copy
import math
import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torch_frame
from relbench.data import RelBenchDataset
from relbench.data.task import TaskType
from relbench.datasets import get_dataset
from relbench.external.graph import (
    get_stype_proposal,
    get_train_table_input,
    make_pkey_fkey_graph,
)
from relbench.external.nn import (
    HeteroEncoder,
    HeteroGraphSAGE,
    HeteroTemporalEncoder,
)
from sentence_transformers import SentenceTransformer
from torch import Tensor
from torch.nn import BCEWithLogitsLoss, L1Loss
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.data import TensorFrame
from tqdm import tqdm

from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import MLP
from torch_geometric.seed import seed_everything
from torch_geometric.typing import EdgeType, NodeType


class GloveTextEmbedding:
    def __init__(self, device: Optional[torch.device] = None):
        self.model = SentenceTransformer(
            "sentence-transformers/average_word_embeddings_glove.6B.300d",
            device=device)

    def __call__(self, sentences: List[str]) -> Tensor:
        return torch.from_numpy(self.model.encode(sentences))


# Stores the informative text columns to retain for each table:
dataset_to_informative_text_cols = {}
dataset_to_informative_text_cols["rel-stackex"] = {
    "postHistory": ["Text"],
    "users": ["AboutMe"],
    "posts": ["Body", "Title", "Tags"],
    "comments": ["Text"],
}

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="rel-stackex")
parser.add_argument("--task", type=str, default="rel-stackex-engage")
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--channels", type=int, default=128)
parser.add_argument("--aggr", type=str, default="sum")
parser.add_argument("--num_neighbors", type=int, default=128)
parser.add_argument("--num_workers", type=int, default=1)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_everything(42)

root_dir = "./data"

# TODO: remove process=True once correct data is uploaded.
dataset: RelBenchDataset = get_dataset(name=args.dataset, process=True)
task = dataset.get_task(args.task)

col_to_stype_dict = get_stype_proposal(dataset.db)
informative_text_cols: Dict = dataset_to_informative_text_cols[args.dataset]
for table_name, stype_dict in col_to_stype_dict.items():
    for col_name, stype in list(stype_dict.items()):
        # Remove text columns except for the informative ones:
        if stype == torch_frame.text_embedded:
            if col_name not in informative_text_cols.get(table_name, []):
                del stype_dict[col_name]

data: HeteroData = make_pkey_fkey_graph(
    dataset.db,
    col_to_stype_dict=col_to_stype_dict,
    text_embedder_cfg=TextEmbedderConfig(
        text_embedder=GloveTextEmbedding(device=device), batch_size=256),
    cache_dir=os.path.join(root_dir, f"{args.dataset}_materialized_cache"),
)

loader_dict: Dict[str, NeighborLoader] = {}
for split, table in [
    ("train", task.train_table),
    ("val", task.val_table),
    ("test", task.test_table),
]:
    table_input = get_train_table_input(table=table, task=task)
    entity_table = table_input.nodes[0]
    loader_dict[split] = NeighborLoader(
        data,
        num_neighbors=[args.num_neighbors, args.num_neighbors],
        time_attr="time",
        input_nodes=table_input.nodes,
        input_time=table_input.time,
        transform=table_input.transform,
        batch_size=args.batch_size,
        shuffle=split == "train",
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )

if task.task_type == TaskType.BINARY_CLASSIFICATION:
    out_channels = 1
    loss_fn = BCEWithLogitsLoss()
    tune_metric = "roc_auc"
    higher_is_better = True
elif task.task_type == TaskType.REGRESSION:
    out_channels = 1
    loss_fn = L1Loss()
    tune_metric = "mae"
    higher_is_better = False


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = HeteroEncoder(
            channels=args.channels,
            node_to_col_names_dict={
                node_type: data[node_type].tf.col_names_dict
                for node_type in data.node_types
            },
            node_to_col_stats=data.col_stats_dict,
        )
        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=[
                node_type for node_type in data.node_types
                if "time" in data[node_type]
            ],
            channels=args.channels,
        )
        self.gnn = HeteroGraphSAGE(
            node_types=data.node_types,
            edge_types=data.edge_types,
            channels=args.channels,
            aggr=args.aggr,
        )
        self.head = MLP(
            args.channels,
            out_channels=out_channels,
            num_layers=1,
        )

    def forward(
        self,
        tf_dict: Dict[NodeType, TensorFrame],
        edge_index_dict: Dict[EdgeType, Tensor],
        seed_time: Tensor,
        time_dict: Dict[NodeType, Tensor],
        batch_dict: Dict[NodeType, Tensor],
        num_sampled_nodes_dict: Dict[NodeType, List[int]],
        num_sampled_edges_dict: Dict[EdgeType, List[int]],
    ) -> Tensor:
        x_dict = self.encoder(tf_dict)

        rel_time_dict = self.temporal_encoder(seed_time, time_dict, batch_dict)
        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time

        x_dict = self.gnn(
            x_dict,
            edge_index_dict,
            num_sampled_nodes_dict,
            num_sampled_edges_dict,
        )

        return self.head(x_dict[entity_table][:seed_time.size(0)])


model = Model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


def train() -> Dict[str, float]:
    model.train()

    loss_accum = count_accum = 0
    for batch in tqdm(loader_dict["train"]):
        batch = batch.to(device)

        optimizer.zero_grad()
        pred = model(
            batch.tf_dict,
            batch.edge_index_dict,
            batch[entity_table].seed_time,
            batch.time_dict,
            batch.batch_dict,
            batch.num_sampled_nodes_dict,
            batch.num_sampled_edges_dict,
        )
        pred = pred.view(-1) if pred.size(1) == 1 else pred
        loss = loss_fn(pred, batch[entity_table].y)
        loss.backward()
        optimizer.step()

        loss_accum += loss.detach().item() * pred.size(0)
        count_accum += pred.size(0)

    return loss_accum / count_accum


@torch.no_grad()
def test(loader: NeighborLoader) -> np.ndarray:
    model.eval()

    pred_list = []
    for batch in tqdm(loader):
        batch = batch.to(device)
        pred = model(
            batch.tf_dict,
            batch.edge_index_dict,
            batch[entity_table].seed_time,
            batch.time_dict,
            batch.batch_dict,
            batch.num_sampled_nodes_dict,
            batch.num_sampled_edges_dict,
        )

        if task.task_type == TaskType.BINARY_CLASSIFICATION:
            pred = torch.sigmoid(pred)

        pred = pred.view(-1) if pred.size(1) == 1 else pred
        pred_list.append(pred.detach().cpu())
    return torch.cat(pred_list, dim=0).numpy()


state_dict = None
best_val_metric = 0 if higher_is_better else math.inf
for epoch in range(1, args.epochs + 1):
    train_loss = train()
    val_pred = test(loader_dict["val"])
    val_metrics = task.evaluate(val_pred, task.val_table)
    print(
        f"Epoch: {epoch:02d}, Train loss: {train_loss}, Val metrics: {val_metrics}"  #noqa
    )

    if (higher_is_better and val_metrics[tune_metric] > best_val_metric) or (
            not higher_is_better
            and val_metrics[tune_metric] < best_val_metric):
        best_val_metric = val_metrics[tune_metric]
        state_dict = copy.deepcopy(model.state_dict())

model.load_state_dict(state_dict)
val_pred = test(loader_dict["val"])
val_metrics = task.evaluate(val_pred, task.val_table)
print(f"Best Val metrics: {val_metrics}")

test_pred = test(loader_dict["test"])
test_metrics = task.evaluate(test_pred)
print(f"Best test metrics: {test_metrics}")
