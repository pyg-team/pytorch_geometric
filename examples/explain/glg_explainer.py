import os
import copy
import time

import torch
from torch_geometric.data import (
    InMemoryDataset,
    Data,
    download_url,
    extract_zip,
    makedirs
)
from torch_geometric.nn import (
    SAGEConv,
    GCNConv,
    global_mean_pool,
    global_add_pool,
    global_max_pool,
    GINConv,
    GATv2Conv,
)
from torch_geometric.explain import GLGExplainer
from torch_geometric.loader import DataLoader
from torch import Tensor, LongTensor
from typing import Union, List

import networkx as nx
import numpy as np

log_models = True
data_path = "./local_explanations/BAMultiShapes"
output_path = "./output"
house = nx.Graph()
house.add_edge(0, 1)
house.add_edge(0, 2)
house.add_edge(2, 1)
house.add_edge(2, 3)
house.add_edge(1, 4)
house.add_edge(4, 3)
grid = nx.grid_2d_graph(3, 3)
wheel = nx.generators.classic.wheel_graph(6)
le_classes_names = [
    "house",
    "grid",
    "wheel",
    "ba",
    "house+grid",
    "house+wheel",
    "wheel+grid",
    "all"
]

if log_models:
    makedirs(f"{output_path}/plots")
    makedirs(f"{output_path}/trained_models")

download_url(
    "https://github.com/steveazzolin/gnn_logic_global_expl/raw/master/"
    "local_explanations/PGExplainer/BAMultiShapes/"
    "GCN/zipped_local_explanations.zip",
    data_path
)
extract_zip(
    f"{data_path}/zipped_local_explanations.zip",
    data_path
)


class LocalExplanationsDataset(InMemoryDataset):
    def __init__(
            self,
            edge_indexes: List[Tensor],
            belonging: Union[Tensor, List],
            y: LongTensor = None,
            le_label: LongTensor = None,
            features: Tensor = None,
            transform=None,
            pre_transform=None,
            pre_filter=None
    ):
        super().__init__("", transform, pre_transform, pre_filter)

        data_list = []
        for i, edge_index in enumerate(edge_indexes):
            num_nodes = edge_index.flatten().max() + 1

            if features is None:
                x = torch.full((num_nodes, 5), 0.1)
            else:
                x = features[i]

            data = Data(
                x=x,
                edge_index=edge_index,
                num_nodes=num_nodes,
                y=y[belonging[i]],  # the class of the original input graph
                le_label=le_label[i],  # the type of local explanation
                le_id=torch.tensor(i, dtype=torch.long),
                graph_id=belonging[i],
            )
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.data, self.slices = self.collate(data_list)


class LEEmbedder(torch.nn.Module):
    r"""Local Explanations Embedder Network"""

    def __init__(
            self,
            num_features,
            activation,
            num_gnn_hidden=20,
            dropout=0.1,
            num_hidden=10,
            num_layers=2,
            backbone="GIN"
    ):
        super().__init__()

        if backbone == "GIN":
            nns = torch.nn.ModuleList([
                torch.nn.Sequential(
                    torch.nn.Linear(
                        num_features if i == 0 else num_gnn_hidden,
                        num_gnn_hidden
                    ),
                    torch.nn.LeakyReLU(),
                    torch.nn.Dropout(dropout)
                ) for i in range(num_layers)]
            )
            self.convs = torch.nn.ModuleList([
                GINConv(nns[i], train_eps=False) for i in range(num_layers)
            ])
        elif backbone == "GAT":
            self.convs = torch.nn.ModuleList([
                GATv2Conv(
                    num_features if i == 0 else num_gnn_hidden,
                    int(num_gnn_hidden / 4),
                    heads=4
                ) for i in range(num_layers)
            ])
        elif backbone == "SAGE":
            self.convs = torch.nn.ModuleList([
                SAGEConv(
                    num_features if i == 0 else num_gnn_hidden,
                    num_gnn_hidden
                ) for i in range(num_layers)
            ])
        elif backbone == "GCN":
            self.convs = torch.nn.ModuleList([
                GCNConv(
                    num_features if i == 0 else num_gnn_hidden,
                    num_gnn_hidden
                ) for i in range(num_layers)
            ])
        else:
            raise ValueError("Backbone not available")

        self.proj = torch.nn.Linear(num_gnn_hidden * 3, num_hidden)
        self.num_layers = num_layers

        if activation == "sigmoid":
            self.actv = torch.nn.Sigmoid()
        elif activation == "tanh":
            self.actv = torch.nn.Tanh()
        elif activation == "leaky":
            self.actv = torch.nn.LeakyReLU()
        elif activation == "lin":
            self.actv = torch.nn.LeakyReLU(negative_slope=1)
        else:
            raise ValueError("Activation not available")

    def forward(self, x, edge_index, batch):
        x = self.get_graph_emb(x, edge_index, batch)
        x = self.actv(self.proj(x))
        return x

    def get_graph_emb(self, x, edge_index, batch):
        x = self.get_emb(x, edge_index)

        x1 = global_mean_pool(x, batch)
        x2 = global_add_pool(x, batch)
        x3 = global_max_pool(x, batch)
        x = torch.cat([x1, x2, x3], dim=-1)
        return x

    def get_emb(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.actv(self.convs[i](x.float(), edge_index))
        return x


def elbow_method(weights, index_stopped=None, min_num_include=7, backup=None):
    sorted_weights = sorted(weights, reverse=True)
    sorted_weights = np.convolve(
        sorted_weights,
        np.ones(min_num_include),
        'valid'
    ) / min_num_include

    stop = np.mean(sorted_weights) if backup is None else backup
    for i in range(len(sorted_weights) - 2):
        if i < min_num_include:
            continue
        if sorted_weights[i - 1] - sorted_weights[i] > 0.0:
            if (sorted_weights[i - 1] - sorted_weights[i] >=
                    40 * (sorted_weights[0] - sorted_weights[i - 2]) / 100
                    + (sorted_weights[0] - sorted_weights[i - 2])):
                stop = sorted_weights[i]
                if index_stopped is not None:
                    index_stopped.append(stop)
                break
    return stop


def assign_class(pattern):
    if len(pattern) == 0:  # BA
        return 3
    elif len(pattern) == 1:  # single motif
        return pattern[0]
    else:
        assert len(pattern) <= 3
        if 0 in pattern and 1 in pattern and 2 in pattern:
            return 7
        elif 0 in pattern and 1 in pattern:
            return 4
        elif 0 in pattern and 2 in pattern:
            return 5
        elif 1 in pattern and 2 in pattern:
            return 6


def label_explanation(G):
    pattern_matched = []
    for i, pattern in enumerate([house, grid, wheel]):
        GM = nx.algorithms.isomorphism.GraphMatcher(G, pattern)
        if GM.subgraph_is_isomorphic():
            pattern_matched.append(i)
    return pattern_matched


def read_bamultishapes(
        split: str,
        min_num_include: bool = 5,
        threshold: bool = None,
):
    r"""Load saved local explanations and apply the elbow method
    to cut the irrelevant edges"""

    adjs, edge_weights, index_stopped = [], [], []
    ori_classes, belonging = [], []
    total_cc_labels, le_classes = [], []

    num_iter = 0
    for split in [split]:
        for c in [1, 0]:
            path = f"{data_path}/{split}/{c}/"
            for pp in os.listdir(path):
                adj = np.load(path + pp, allow_pickle=True)

                cut = elbow_method(
                    np.triu(adj).flatten(),
                    index_stopped,
                    min_num_include
                ) if threshold is None else threshold
                masked = copy.deepcopy(adj)
                masked[masked <= cut] = 0
                masked[masked > cut] = 1
                G = nx.from_numpy_matrix(masked)

                added = 0
                gnn_pred = int(pp.split("_")[0])
                total_cc_labels.append([])
                for cc in nx.connected_components(G):
                    if len(cc) > 2:
                        G1 = G.subgraph(cc)

                        # if not a trivial graph
                        if not nx.diameter(G1) == len(G1.edges()):
                            cc_lbl = label_explanation(G1)
                            added += 1
                            adjs.append(nx.to_numpy_matrix(G1))
                            edge_weights.append(
                                nx.get_edge_attributes(G1, "weight")
                            )
                            belonging.append(num_iter)
                            le_classes.append(assign_class(cc_lbl))
                if added:
                    num_iter += 1
                    ori_classes.append(gnn_pred)
    belonging = GLGExplainer.normalize_belonging(belonging)
    return (
        adjs,
        edge_weights,
        torch.tensor(ori_classes, dtype=torch.long),
        torch.tensor(belonging),
        torch.tensor(le_classes, dtype=torch.long)
    )

##
# read the saved local explanations
##


(
    adjs_train,
    edge_weights_train,
    task_y_train,
    belonging_train,
    y_train
) = read_bamultishapes(split="TRAIN")

(
    adjs_val,
    edge_weights_val,
    task_y_val,
    belonging_val,
    y_val
) = read_bamultishapes(split="VAL")

(
    adjs_test,
    edge_weights_test,
    task_y_test,
    belonging_test,
    y_test
) = read_bamultishapes(split="TEST")

##
# extract edge index from adj. matrix and create Datasets
##
edge_index_train = [torch.tensor(a).nonzero().t() for a in adjs_train]
edge_index_val = [torch.tensor(a).nonzero().t() for a in adjs_val]
edge_index_test = [torch.tensor(a).nonzero().t() for a in adjs_test]

dataset_train = LocalExplanationsDataset(
    edge_index_train,
    y=task_y_train,
    belonging=belonging_train,
    le_label=y_train
)
dataset_val = LocalExplanationsDataset(
    edge_index_val,
    y=task_y_val,
    belonging=belonging_val,
    le_label=y_val
)
dataset_test = LocalExplanationsDataset(
    edge_index_test,
    y=task_y_test,
    belonging=belonging_test,
    le_label=y_test
)

##
# create DataLoader with the provided custom sampler
##

train_group_loader = DataLoader(
    dataset_train,
    batch_sampler=GLGExplainer.get_sampler(
        num_input_graphs=128,
        drop_last=False,
        belonging=belonging_train
    )
)
val_group_loader = DataLoader(
    dataset_val,
    batch_sampler=GLGExplainer.get_sampler(
        num_input_graphs=128,
        drop_last=False,
        belonging=belonging_val
    )
)
test_group_loader = DataLoader(
    dataset_test,
    batch_sampler=GLGExplainer.get_sampler(
        num_input_graphs=128,
        drop_last=False,
        belonging=belonging_test
    )
)


##
# init the models
##
torch.manual_seed(43)
le_model = LEEmbedder(
    num_features=5,
    activation="leaky",
    num_hidden=10
)
expl = GLGExplainer(
    le_embedder=le_model,
    num_classes=2,
    num_prototypes=6
)

print(expl)


##
# train the explainer
##

start_time = time.time()
best_val_loss = np.inf
for epoch in range(1, 2000):
    train_metrics = expl.do_epoch(train_group_loader, train=True)
    val_metrics = expl.do_epoch(val_group_loader, train=False)

    if epoch % 40 == 0:
        train_log_metrics = expl.inspect(
            train_group_loader,
            plot=True,
            plot_path=f"{output_path}/plots/{epoch}.png",
            le_classes_names=le_classes_names,
            update_global_explanation=True
        )
        val_log_metrics = expl.inspect(val_group_loader, plot=False)

        print(
            f"Concept Purity: {train_log_metrics['concept_purity']}"
            "+-"
            f"{train_log_metrics['concept_purity_std']}"
        )
        print(f"Fidelity: {train_log_metrics['fidelity']}")
        print(f"Formula Accuracy: {train_log_metrics['formula_accuracy']}\n")

        ##
        # print logic explanations
        ##
        for c, e in enumerate(train_log_metrics['explanations'].explanations):
            print(f"Class {c}")
            print(e.cwa())

    if val_metrics["loss"] < best_val_loss and log_models:
        best_val_loss = val_metrics["loss"]
        torch.save(
            expl.state_dict(),
            f"{output_path}/trained_models/GLGExplainer_epoch_{epoch}.pt"
        )

    print(
        f'{epoch:3d}: Loss: {train_metrics["loss"]:.5f},'
        f'LEN: {train_metrics["len_loss"]:2f},'
        f'Fid: {train_metrics["fidelity"]:.2f},'
        f'V. Fid: {val_metrics["fidelity"]:.2f},'
        f'V. Loss: {val_metrics["loss"]:.5f},'
        f'V. LEN {val_metrics["len_loss"]:.3f}'
    )

    if expl.early_stopping.on_epoch_end(epoch, val_metrics["loss"]):
        print("Early Stopping")
        print(f"Loading model at epoch {expl.early_stopping.best_epoch}")
        if log_models:
            expl.load_state_dict(
                train_group_loader,
                **torch.load(
                    f"{output_path}/trained_models/"
                    f"GLGExplainer_epoch_{expl.early_stopping.best_epoch}.pt"
                )
            )
        else:
            print("Model not loaded")
        break

print(f"Best epoch: {expl.early_stopping.best_epoch}")
print(f"Trained lasted for {round(time.time() - start_time)} seconds")
print("")

##
# extract best embedding and best explanations
##
best_epoch_metrics = expl.inspect(
    train_group_loader,
    plot=True,
    plot_path=f"{output_path}/plots/best_epoch.png",
    le_classes_names=le_classes_names,
    update_global_explanation=True
)

for c, e in enumerate(best_epoch_metrics['explanations'].explanations):
    print(f"Class {c}")
    print(e.cwa())
