"""This is a demo of Relational Deep Learning code for RelBench dataset.
For more details on RelBench and Relational Deep learning, please refer to
https://arxiv.org/abs/2407.20060 and
https://raw.githubusercontent.com/mlresearch/v235/main/assets/fey24a/fey24a.pdf
This is NOT the official code for the experiments in these papers.
To exactly reproduce the experimental results, from the linked papers, please
see https://github.com/snap-stanford/relbench or
https://github.com/kumo-ai/ContextGNN for link prediction.
"""
import argparse
import copy
import math
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch_frame
from relbench.base import EntityTask, TaskType
from relbench.datasets import get_dataset
from relbench.modeling.graph import (
    get_node_train_table_input,
    make_pkey_fkey_graph,
)
from relbench.modeling.utils import get_stype_proposal
from relbench.tasks import get_task
from sentence_transformers import SentenceTransformer
from torch import Tensor
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.data.stats import StatType
from torch_frame.nn.models import ResNet
from tqdm import tqdm

from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import (
    MLP,
    HeteroConv,
    LayerNorm,
    PositionalEncoding,
    SAGEConv,
)
from torch_geometric.seed import seed_everything
from torch_geometric.typing import EdgeType, NodeType

seed_everything(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root_dir = "./data"
print("Using device: ", device)

args = argparse.ArgumentParser()

# Dataset argument
args.add_argument(
    "--dataset", type=str, default="rel-f1", choices=[
        "rel-stack", "rel-amazon", "rel-trial", "rel-f1", "rel-hm",
        "rel-event", "rel-avito"
    ])
known_args, _ = args.parse_known_args()

# task argument choices based are based on the dataset, this example only
# supports node level tasks
if known_args.dataset == "rel-f1":
    args.add_argument("--task", type=str, default="driver-position",
                      choices=["driver-position", "driver-dnf", "driver-top3"])
elif known_args.dataset == "rel-amazon":
    args.add_argument(
        "--task", type=str, default="user-churn",
        choices=["user-churn", "item-churn", "user-ltv", "item-ltv"])
elif known_args.dataset == "rel-hm":
    args.add_argument("--task", type=str, default="user-churn",
                      choices=["user-churn", "item-sales"])
elif known_args.dataset == "rel-stack":
    args.add_argument("--task", type=str, default="user-engagement",
                      choices=["user-engagement", "user-badge", "post-votes"])
elif known_args.dataset == "rel-trial":
    args.add_argument(
        "--task", type=str, default="study-outcome",
        choices=["study-outcome", "study-adverse", "site-success"])
elif known_args.dataset == "rel-event":
    args.add_argument(
        "--task", type=str, default="user-repeat",
        choices=["user-repeat", "user-ignore", "user-attendance"])
elif known_args.dataset == "rel-avito":
    args.add_argument("--task", type=str, default="user-visits",
                      choices=["user-visits", "user-clicks", "ad-ctr"])

# Data loader arguments
args.add_argument("--batch_size", type=int, default=512)
args.add_argument("--temporal_strategy", type=str, default="uniform",
                  choices=["uniform", "last"])
args.add_argument("--num_workers", type=int, default=0)
args.add_argument("--persistent_workers", type=bool, default=False)

# Model parameters
args.add_argument("--num_neighbors", type=list, default=[128, 128])
args.add_argument("--channels", type=int, default=128)
args.add_argument("--out_channels", type=int, default=1)
args.add_argument("--aggr", type=str, default="sum")
args.add_argument("--norm", type=str, default="batch_norm")

# Training parameters
args.add_argument("--epochs", type=int, default=10)
args.add_argument("--learning_rate", type=float, default=0.005)

args = args.parse_args()


class GloveTextEmbedding:
    """GloveTextEmbedding based on SentenceTransformer."""
    def __init__(self, device: Optional[torch.device] = None):
        self.model = SentenceTransformer(
            "sentence-transformers/average_word_embeddings_glove.6B.300d",
            device=device,
        )

    def __call__(self, sentences: List[str]) -> Tensor:
        return torch.from_numpy(self.model.encode(sentences))


class HeteroEncoder(torch.nn.Module):
    r"""HeteroEncoder based on PyTorch Frame.

    A heterogeneous encoder that processes different node types using PyTorch
    Frame models. For each node type, it creates a separate encoder model
    that processes the node features according to their data types
    (categorical, numerical, etc).

    Args:
        channels (int): The output channels for each node type.
        node_to_col_names_dict
            (Dict[NodeType, Dict[torch_frame.stype, List[str]]]):
            A dictionary mapping from node type to column names dictionary
            compatible to PyTorch Frame.
        node_to_col_stats (Dict[NodeType, Dict[str, Dict[StatType, Any]]]):
            A dictionary containing statistics for each column
            in each node type. Used for feature normalization and encoding.
        torch_frame_model_cls: Model class for PyTorch Frame. The class object
            takes :class:`TensorFrame` object as input and outputs
            :obj:`channels`-dimensional embeddings. Default to
            :class:`torch_frame.nn.ResNet`.
        torch_frame_model_kwargs (Dict[str, Any]): Keyword arguments for
            :class:`torch_frame_model_cls` class. Default keyword argument is
            set specific for :class:`torch_frame.nn.ResNet`. Expect it to
            be changed for different :class:`torch_frame_model_cls`.
        default_stype_encoder_cls_kwargs (Dict[torch_frame.stype, Any]):
            A dictionary mapping from :obj:`torch_frame.stype` object into a
            tuple specifying :class:`torch_frame.nn.StypeEncoder` class and its
            keyword arguments :obj:`kwargs`.
    """
    def __init__(
        self,
        channels: int,
        node_to_col_names_dict: Dict[NodeType, Dict[torch_frame.stype,
                                                    List[str]]],
        node_to_col_stats: Dict[NodeType, Dict[str, Dict[StatType, Any]]],
        torch_frame_model_cls=ResNet,
        torch_frame_model_kwargs: Dict[str, Any] = {
            "channels": 128,
            "num_layers": 4,
        },
        default_stype_encoder_cls_kwargs: Dict[torch_frame.stype, Any] = {
            torch_frame.categorical: (torch_frame.nn.EmbeddingEncoder, {}),
            torch_frame.numerical: (torch_frame.nn.LinearEncoder, {}),
            torch_frame.multicategorical: (
                torch_frame.nn.MultiCategoricalEmbeddingEncoder,
                {},
            ),
            torch_frame.embedding: (torch_frame.nn.LinearEmbeddingEncoder, {}),
            torch_frame.timestamp: (torch_frame.nn.TimestampEncoder, {}),
        },
    ):
        super().__init__()

        self.encoders = torch.nn.ModuleDict()

        for node_type in node_to_col_names_dict.keys():
            stype_encoder_dict = {
                stype:
                default_stype_encoder_cls_kwargs[stype][0](
                    **default_stype_encoder_cls_kwargs[stype][1])
                for stype in node_to_col_names_dict[node_type].keys()
            }
            torch_frame_model = torch_frame_model_cls(
                **torch_frame_model_kwargs,
                out_channels=channels,
                col_stats=node_to_col_stats[node_type],
                col_names_dict=node_to_col_names_dict[node_type],
                stype_encoder_dict=stype_encoder_dict,
            )
            self.encoders[node_type] = torch_frame_model

    def reset_parameters(self):
        """Reset the parameters of all encoder models."""
        for encoder in self.encoders.values():
            encoder.reset_parameters()

    def forward(
        self,
        tf_dict: Dict[NodeType, torch_frame.TensorFrame],
    ) -> Dict[NodeType, Tensor]:
        """Forward pass of the heterogeneous encoder.

        Args:
            tf_dict (Dict[NodeType, torch_frame.TensorFrame]):
                Dictionary mapping node types to their corresponding
                TensorFrame objects containing the node features.

        Returns:
            Dict[NodeType, Tensor]: Dictionary mapping node
            types to their encoded representations. Each tensor
            has shape [num_nodes, channels].
        """
        x_dict = {
            node_type: self.encoders[node_type](tf)
            for node_type, tf in tf_dict.items()
        }
        return x_dict


class HeteroTemporalEncoder(torch.nn.Module):
    """HeteroTemporalEncoder class that uses PositionalEncoding to encode
    temporal information for heterogeneous graphs.

    This encoder computes relative time embeddings between a seed time and
    node timestamps, converting the time differences from seconds to days.
    It applies positional encoding followed by a linear transformation for
    each node type.

    Args:
        node_types (List[NodeType]):
            List of node types in the heterogeneous graph
        channels (int):
            Number of channels/dimensions for the encoded embeddings

    Example:
        >>> encoder = HeteroTemporalEncoder(['user', 'item'], channels=64)
        >>> seed_time = torch.tensor([1000])  # Reference timestamp
        >>> time_dict = {'user': torch.tensor([800, 900]),
        >>>             'item': torch.tensor([700, 850])}
        >>> batch_dict = {'user': torch.tensor([0, 0]),
        >>>              'item': torch.tensor([0, 0])}
        >>> out_dict = encoder(seed_time, time_dict, batch_dict)
        >>> print(out_dict['user'].shape)  # torch.Size([2, 64])
    """
    def __init__(self, node_types: List[NodeType], channels: int):
        super().__init__()

        self.encoder_dict = torch.nn.ModuleDict({
            node_type:
            PositionalEncoding(channels)
            for node_type in node_types
        })
        self.lin_dict = torch.nn.ModuleDict({
            node_type:
            torch.nn.Linear(channels, channels)
            for node_type in node_types
        })

    def reset_parameters(self):
        """Reset the parameters of all encoders and linear layers."""
        for encoder in self.encoder_dict.values():
            encoder.reset_parameters()
        for lin in self.lin_dict.values():
            lin.reset_parameters()

    def forward(
        self,
        seed_time: Tensor,
        time_dict: Dict[NodeType, Tensor],
        batch_dict: Dict[NodeType, Tensor],
    ) -> Dict[NodeType, Tensor]:
        """Forward pass of the temporal encoder.

        Args:
            seed_time (Tensor):
                Reference timestamps for computing relative times
            time_dict (Dict[NodeType, Tensor]):
                Dictionary mapping node types to their timestamps
            batch_dict (Dict[NodeType, Tensor]):
                Dictionary mapping node types to batch assignments

        Returns:
            Dict[NodeType, Tensor]:
                Dictionary mapping node types to their temporal embeddings
        """
        out_dict: Dict[NodeType, Tensor] = {}

        for node_type, time in time_dict.items():
            rel_time = seed_time[batch_dict[node_type]] - time
            rel_time = rel_time / (60 * 60 * 24)  # Convert seconds to days.

            x = self.encoder_dict[node_type](rel_time)
            x = self.lin_dict[node_type](x)
            out_dict[node_type] = x

        return out_dict


class HeteroGraphSAGE(torch.nn.Module):
    """Heterogeneous GraphSAGE model with layer normalization.

    This model implements a heterogeneous version of GraphSAGE
    that operates on multiple node and edge types. Each layer
    consists of a heterogeneous graph convolution followed by
    layer normalization and ReLU activation.

    Args:
        node_types (List[NodeType]): List of node types in the graph
        edge_types (List[EdgeType]): List of edge types in the graph
        channels (int): Number of channels/features
        aggr (str, optional): Node aggregation scheme. Defaults to "mean"
        num_layers (int, optional):
            Number of graph convolution layers. Defaults to 2

    Example:
        >>> model = HeteroGraphSAGE(
        >>>     node_types=['user', 'item'],
        >>>     edge_types=[('user', 'rates', 'item')],
        >>>     channels=64)
        >>> out_dict = model(x_dict, edge_index_dict)
    """
    def __init__(
        self,
        node_types: List[NodeType],
        edge_types: List[EdgeType],
        channels: int,
        aggr: str = "mean",
        num_layers: int = 2,
    ):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    edge_type: SAGEConv(
                        (channels, channels), channels, aggr=aggr)
                    for edge_type in edge_types
                },
                aggr="sum",
            )
            self.convs.append(conv)

        self.norms = torch.nn.ModuleList()
        for _ in range(num_layers):
            norm_dict = torch.nn.ModuleDict()
            for node_type in node_types:
                norm_dict[node_type] = LayerNorm(channels, mode="node")
            self.norms.append(norm_dict)

    def reset_parameters(self):
        """Reset the parameters of all convolution and normalization layers."""
        for conv in self.convs:
            conv.reset_parameters()
        for norm_dict in self.norms:
            for norm in norm_dict.values():
                norm.reset_parameters()

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[NodeType, Tensor],
    ) -> Dict[NodeType, Tensor]:
        """Forward pass of the heterogeneous GraphSAGE model.

        Args:
            x_dict (Dict[NodeType, Tensor]): Node feature dictionary
            edge_index_dict (Dict[NodeType, Tensor]): Edge index dictionary

        Returns:
            Dict[NodeType, Tensor]: Updated node features after message passing
        """
        for _, (conv, norm_dict) in enumerate(zip(self.convs, self.norms)):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: norm_dict[key](x) for key, x in x_dict.items()}
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        return x_dict


class Model(torch.nn.Module):
    """A heterogeneous graph neural network model for temporal graph learning.

    This model consists of:
    1. A heterogeneous feature encoder for node attributes
    2. A temporal encoder for handling time information
    3. A heterogeneous GraphSAGE model for message passing
    4. An MLP head for final predictions

    Args:
        data (HeteroData): The heterogeneous graph data object
        col_stats_dict (Dict[str, Dict[str, Dict[StatType, Any]]]):
            Statistics of node features
        num_layers (int): Number of GNN layers
        channels (int): Hidden dimension size
        out_channels (int): Output dimension size
        aggr (str): Aggregation method for GNN
        norm (str): Normalization method for MLP
    """
    def __init__(
        self,
        data: HeteroData,
        col_stats_dict: Dict[str, Dict[str, Dict[StatType, Any]]],
        num_layers: int,
        channels: int,
        out_channels: int,
        aggr: str,
        norm: str,
    ):
        super().__init__()

        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={
                node_type: data[node_type].tf.col_names_dict
                for node_type in data.node_types
            },
            node_to_col_stats=col_stats_dict,
        )
        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=[
                node_type for node_type in data.node_types
                if "time" in data[node_type]
            ],
            channels=channels,
        )
        self.gnn = HeteroGraphSAGE(
            node_types=data.node_types,
            edge_types=data.edge_types,
            channels=channels,
            aggr=aggr,
            num_layers=num_layers,
        )
        self.head = MLP(
            channels,
            out_channels=out_channels,
            norm=norm,
            num_layers=1,
        )

        self.reset_parameters()

    def reset_parameters(self):
        """Reset the parameters of all model components."""
        self.encoder.reset_parameters()
        self.temporal_encoder.reset_parameters()
        self.gnn.reset_parameters()
        self.head.reset_parameters()

    def forward(
        self,
        batch: HeteroData,
        entity_table: NodeType,
    ) -> Tensor:
        """Forward pass of the model.

        Steps:
            1. Get seed time from entity table
            2. Encode node features using HeteroEncoder
            3. Encode temporal features using HeteroTemporalEncoder
            4. Add temporal embeddings to node features
            5. Apply graph neural network (HeteroGraphSAGE)
            6. Apply final MLP head to target node embeddings

        Args:
            batch (HeteroData): Batch of heterogeneous graph data
            entity_table (NodeType): The target node type for prediction

        Returns:
            Tensor: Predictions for nodes in the entity table
        """
        seed_time = batch[entity_table].seed_time
        x_dict = self.encoder(batch.tf_dict)

        rel_time_dict = self.temporal_encoder(seed_time, batch.time_dict,
                                              batch.batch_dict)

        for node_type, rel_time in rel_time_dict.items():
            x_dict[node_type] = x_dict[node_type] + rel_time

        x_dict = self.gnn(
            x_dict,
            batch.edge_index_dict,
        )

        return self.head(x_dict[entity_table][:seed_time.size(0)])


def get_task_type_params(task):
    """Get task-specific optimization parameters based on task type."""
    if task.task_type == TaskType.REGRESSION:
        out_channels = 1
        loss_fn = torch.nn.L1Loss()
        tune_metric = "mae"
        higher_is_better = False
    elif task.task_type == TaskType.BINARY_CLASSIFICATION:
        out_channels = 1
        loss_fn = torch.nn.BCEWithLogitsLoss()
        tune_metric = "roc_auc"
        higher_is_better = True
    else:
        raise ValueError(f"Unsupported task type: {task.task_type}")

    return out_channels, loss_fn, tune_metric, higher_is_better


def train(
    model: Model,
    loader_dict: Dict[str, NeighborLoader],
    task: EntityTask,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
) -> float:
    model.train()

    loss_accum = count_accum = 0
    for batch in tqdm(loader_dict["train"]):
        batch = batch.to(device)

        optimizer.zero_grad()
        pred = model(
            batch,
            task.entity_table,
        )
        pred = pred.view(-1) if pred.size(1) == 1 else pred

        # Get the target column name from the task
        loss = loss_fn(pred.float(), batch[task.entity_table].y.float())
        loss.backward()
        optimizer.step()

        loss_accum += loss.detach().item() * pred.size(0)
        count_accum += pred.size(0)

    return loss_accum / count_accum


@torch.no_grad()
def test(loader: NeighborLoader, model: Model, task: EntityTask) -> np.ndarray:
    model.eval()

    pred_list = []
    for batch in loader:
        batch = batch.to(device)
        pred = model(
            batch,
            task.entity_table,
        )
        pred = pred.view(-1) if pred.size(1) == 1 else pred
        pred_list.append(pred.detach().cpu())
    return torch.cat(pred_list, dim=0).numpy()


if __name__ == "__main__":
    """ Prepare the data """

    # Load the dataset
    print("Loading dataset...")
    dataset = get_dataset(name=args.dataset, download=True)

    # Load the task
    print("Loading task...")
    task = get_task(dataset_name=args.dataset, task_name=args.task,
                    download=True)

    # Print task information for debugging
    print(f"Task type: {task.task_type}")
    print(f"Target column: {task.target_col}")
    print(f"Entity table: {task.entity_table}")

    # get the column to stype dictionary
    print("Getting column to stype dictionary...")
    db = dataset.get_db()
    col_to_stype_dict = get_stype_proposal(db)
    print("Column to stype dictionary: ", col_to_stype_dict)

    # define the text embedder
    print("Defining text embedder...")
    text_embedder_cfg = TextEmbedderConfig(
        text_embedder=GloveTextEmbedding(device=device), batch_size=256)

    # Transform the dataset into a HeteroData object with torch_frame features
    print("Transforming dataset into HeteroData object...")
    data, col_stats_dict = make_pkey_fkey_graph(
        db,
        col_to_stype_dict=col_to_stype_dict,  # speficied column types
        text_embedder_cfg=text_embedder_cfg,  # our chosen text encoder
        cache_dir=os.path.join(
            root_dir, f"{args.dataset}_{args.task}_materialized_cache"
        ),  # store materialized graph for convenience
    )
    """ Prepare data loaders """
    print("Preparing data loaders...")
    loader_dict = {}

    # load the training, validation, and test tables
    train_table = task.get_table("train")
    val_table = task.get_table("val")
    test_table = task.get_table("test")

    # Print table information for debugging
    print(f"Train table columns: {train_table.df.columns}")
    print(f"Val table columns: {val_table.df.columns}")
    print(f"Test table columns: {test_table.df.columns}")

    for split, table in [
        ("train", train_table),
        ("val", val_table),
        ("test", test_table),
    ]:
        table_input = get_node_train_table_input(
            table=table,
            task=task,
        )
        entity_table = table_input.nodes[0]

        # Create a dictionary mapping edge types to number of neighbors
        num_neighbors_dict = {
            edge_type: args.num_neighbors  # [num_neighbors for each direction]
            for edge_type in data.edge_types
        }

        loader_dict[split] = NeighborLoader(
            data=data,
            num_neighbors=num_neighbors_dict,
            input_nodes=table_input.nodes,
            input_time=table_input.time,
            time_attr="time",
            transform=table_input.transform,
            batch_size=args.batch_size,
            temporal_strategy=args.temporal_strategy,
            shuffle=split == "train",
            num_workers=args.num_workers,
            persistent_workers=args.persistent_workers,
        )

    # Get task-specific parameters
    print("Getting task-specific parameters...")
    (out_channels, loss_fn, tune_metric,
     higher_is_better) = get_task_type_params(task)
    print("out_channels: ", out_channels)
    print("loss_fn: ", loss_fn)
    print("tune_metric: ", tune_metric)
    print("higher_is_better: ", higher_is_better)

    # Define the model
    print("Initializing the model...")
    model = Model(
        data=data,
        col_stats_dict=col_stats_dict,
        num_layers=len(args.num_neighbors),
        channels=args.channels,
        out_channels=args.out_channels,
        aggr=args.aggr,
        norm=args.norm,
    ).to(device)

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    """ Train the model """
    print("Training the model...")
    state_dict = None
    best_val_metric = -math.inf if higher_is_better else math.inf
    for epoch in range(1, args.epochs + 1):
        train_loss = train(
            model=model,
            loader_dict=loader_dict,
            task=task,
            optimizer=optimizer,
            loss_fn=loss_fn,
        )
        val_pred = test(
            loader=loader_dict["val"],
            model=model,
            task=task,
        )
        val_metrics = task.evaluate(val_pred, val_table)
        print(f"Epoch: {epoch:02d}, "
              f"Train loss: {train_loss}, "
              f"Val metrics: {val_metrics}")

        if (higher_is_better and val_metrics[tune_metric] > best_val_metric
            ) or (not higher_is_better
                  and val_metrics[tune_metric] < best_val_metric):
            best_val_metric = val_metrics[tune_metric]
            state_dict = copy.deepcopy(model.state_dict())
    """ Evaluate the model """

    # Load the best model state dictionary
    print("Loading the best model state dictionary...")
    model.load_state_dict(state_dict)
    val_pred = test(
        loader=loader_dict["val"],
        model=model,
        task=task,
    )

    val_metrics = task.evaluate(val_pred, val_table)
    print(f"Validation metrics: {val_metrics}")

    # Test the model on the test set
    test_pred = test(
        loader=loader_dict["test"],
        model=model,
        task=task,
    )
    test_metrics = task.evaluate(test_pred)
    print(f"Test metrics: {test_metrics}")
