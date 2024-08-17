import os
from itertools import chain
from typing import Iterator, List, Tuple, no_type_check

import numpy as np
import torch
from tqdm import tqdm

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    LargeGraphIndexer,
    TripletLike,
    get_features_for_triplets_groups,
)
from torch_geometric.data.large_graph_indexer import EDGE_RELATION
from torch_geometric.nn.nlp import SentenceTransformer

try:
    from pandas import DataFrame
    WITH_PANDAS = True
except ImportError:
    DataFrame = None
    WITH_PANDAS = False

try:
    from pcst_fast import pcst_fast

    WITH_PCST = True
except ImportError:
    WITH_PCST = False

try:
    import datasets

    WITH_DATASETS = True
except ImportError:
    WITH_DATASETS = False


@no_type_check
def retrieval_via_pcst(
    graph: Data,
    q_emb: torch.Tensor,
    textual_nodes: DataFrame,
    textual_edges: DataFrame,
    topk: int = 3,
    topk_e: int = 3,
    cost_e: float = 0.5,
    save_idx: bool = False,
    override: bool = False,
) -> Tuple[Data, str]:
    c = 0.01
    if len(textual_nodes) == 0 or len(textual_edges) == 0 or override:
        desc = (textual_nodes.to_csv(index=False) +
                "\n" + textual_edges.to_csv(
                    index=False, columns=["src", "edge_attr", "dst"]))
        graph = Data(
            x=graph.x,
            edge_index=graph.edge_index,
            edge_attr=graph.edge_attr,
            num_nodes=graph.num_nodes,
        )
        return graph, desc

    root = -1  # unrooted
    num_clusters = 1
    pruning = "gw"
    verbosity_level = 0
    if topk > 0:
        n_prizes = torch.nn.CosineSimilarity(dim=-1)(q_emb, graph.x)
        topk = min(topk, graph.num_nodes)
        _, topk_n_indices = torch.topk(n_prizes, topk, largest=True)

        n_prizes = torch.zeros_like(n_prizes)
        n_prizes[topk_n_indices] = torch.arange(topk, 0, -1).float()
    else:
        n_prizes = torch.zeros(graph.num_nodes)

    if topk_e > 0:
        e_prizes = torch.nn.CosineSimilarity(dim=-1)(q_emb, graph.edge_attr)
        topk_e = min(topk_e, e_prizes.unique().size(0))

        topk_e_values, _ = torch.topk(e_prizes.unique(), topk_e, largest=True)
        e_prizes[e_prizes < topk_e_values[-1]] = 0.0
        last_topk_e_value = topk_e
        for k in range(topk_e):
            indices = e_prizes == topk_e_values[k]
            value = min((topk_e - k) / sum(indices), last_topk_e_value - c)
            e_prizes[indices] = value
            last_topk_e_value = value
        # cost_e = max(min(cost_e, e_prizes.max().item()-c), 0)
    else:
        e_prizes = torch.zeros(graph.num_edges)

    costs = []
    edges = []
    virtual_n_prizes = []
    virtual_edges = []
    virtual_costs = []
    mapping_n = {}
    mapping_e = {}
    for i, (src, dst) in enumerate(graph.edge_index.T.numpy()):
        prize_e = e_prizes[i]
        if prize_e <= cost_e:
            mapping_e[len(edges)] = i
            edges.append((src, dst))
            costs.append(cost_e - prize_e)
        else:
            virtual_node_id = graph.num_nodes + len(virtual_n_prizes)
            mapping_n[virtual_node_id] = i
            virtual_edges.append((src, virtual_node_id))
            virtual_edges.append((virtual_node_id, dst))
            virtual_costs.append(0)
            virtual_costs.append(0)
            virtual_n_prizes.append(prize_e - cost_e)

    prizes = np.concatenate([n_prizes, np.array(virtual_n_prizes)])
    num_edges = len(edges)
    if len(virtual_costs) > 0:
        costs = np.array(costs + virtual_costs)
        edges = np.array(edges + virtual_edges)

    vertices, edges = pcst_fast(edges, prizes, costs, root, num_clusters,
                                pruning, verbosity_level)

    selected_nodes = vertices[vertices < graph.num_nodes]
    selected_edges = [mapping_e[e] for e in edges if e < num_edges]
    virtual_vertices = vertices[vertices >= graph.num_nodes]
    if len(virtual_vertices) > 0:
        virtual_vertices = vertices[vertices >= graph.num_nodes]
        virtual_edges = [mapping_n[i] for i in virtual_vertices]
        selected_edges = np.array(selected_edges + virtual_edges)

    edge_index = graph.edge_index[:, selected_edges]
    selected_nodes = np.unique(
        np.concatenate(
            [selected_nodes, edge_index[0].numpy(), edge_index[1].numpy()]))

    n = textual_nodes.iloc[selected_nodes]
    e = textual_edges.iloc[selected_edges]
    desc = (n.to_csv(index=False) + "\n" +
            e.to_csv(index=False, columns=["src", "edge_attr", "dst"]))

    mapping = {n: i for i, n in enumerate(selected_nodes.tolist())}

    x = graph.x[selected_nodes]
    edge_attr = graph.edge_attr[selected_edges]
    src = [mapping[i] for i in edge_index[0].tolist()]
    dst = [mapping[i] for i in edge_index[1].tolist()]
    edge_index = torch.LongTensor([src, dst])
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                num_nodes=len(selected_nodes))

    # HACK Added so that the subset of nodes and edges selected can be tracked
    if save_idx:
        data['node_idx'] = np.array(graph.node_idx)[selected_nodes]
        data['edge_idx'] = np.array(graph.edge_idx)[selected_edges]

    return data, desc


def preprocess_triplet(triplet: TripletLike) -> TripletLike:
    h, r, t = triplet
    return str(h).lower(), str(r), str(t).lower()


class UpdatedWebQSPDataset(InMemoryDataset):
    def __init__(
        self,
        root: str = "",
        force_reload: bool = False,
        whole_graph_retrieval: bool = False,
        limit: int = -1,
        override: bool = False,
    ) -> None:
        self.limit = limit
        self.whole_graph_retrieval = whole_graph_retrieval
        self.override = override
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(root, None, None, force_reload=force_reload)
        self._load_raw_data()
        self.load(self.processed_paths[0])

    def _check_dependencies(self) -> None:
        missing_str_list = []
        if not WITH_PCST:
            missing_str_list.append('pcst_fast')
        if not WITH_DATASETS:
            missing_str_list.append('datasets')
        if not WITH_PANDAS:
            missing_str_list.append('pandas')
        if len(missing_str_list) > 0:
            missing_str = ' '.join(missing_str_list)
            error_out = f"`pip install {missing_str}` to use this dataset."
            raise ImportError(error_out)

    @property
    def raw_file_names(self) -> List[str]:
        return ["raw_data", "split_idxs"]

    @property
    def processed_file_names(self) -> List[str]:
        return [
            "list_of_graphs.pt",
            "pre_filter.pt",
            "pre_transform.pt",
            "large_graph_indexer",
        ]

    def _save_raw_data(self) -> None:
        self.raw_dataset.save_to_disk(self.raw_paths[0])
        torch.save(self.split_idxs, self.raw_paths[1])

    def _load_raw_data(self) -> None:
        if not hasattr(self, "raw_dataset"):
            self.raw_dataset = datasets.load_from_disk(self.raw_paths[0])
        if not hasattr(self, "split_idxs"):
            self.split_idxs = torch.load(self.raw_paths[1])

    def download(self) -> None:
        dataset = datasets.load_dataset("rmanluo/RoG-webqsp")
        self.raw_dataset = datasets.concatenate_datasets(
            [dataset["train"], dataset["validation"], dataset["test"]])
        self.split_idxs = {
            "train":
            torch.arange(len(dataset["train"])),
            "val":
            torch.arange(len(dataset["validation"])) + len(dataset["train"]),
            "test":
            torch.arange(len(dataset["test"])) + len(dataset["train"]) +
            len(dataset["validation"]),
        }

        if self.limit >= 0:
            self.raw_dataset = self.raw_dataset.select(range(self.limit))
            self.split_idxs = {
                "train":
                torch.arange(self.limit // 2),
                "val":
                torch.arange(self.limit // 4) + self.limit // 2,
                "test":
                torch.arange(self.limit // 4) + self.limit // 2 +
                self.limit // 4,
            }
        self._save_raw_data()

    def _get_trips(self) -> Iterator[TripletLike]:
        return chain.from_iterable(
            iter(ds["graph"]) for ds in self.raw_dataset)

    def _build_graph(self) -> None:
        trips = self._get_trips()
        self.indexer: LargeGraphIndexer = LargeGraphIndexer.from_triplets(
            trips, pre_transform=preprocess_triplet)

        # Nodes:
        nodes = self.indexer.get_unique_node_features()
        x = self.model.encode(nodes, batch_size=256)  # type: ignore
        self.indexer.add_node_feature(new_feature_name="x", new_feature_vals=x)

        # Edges:
        edges = self.indexer.get_unique_edge_features(
            feature_name=EDGE_RELATION)
        edge_attr = self.model.encode(edges, batch_size=256)  # type: ignore
        self.indexer.add_edge_feature(
            new_feature_name="edge_attr",
            new_feature_vals=edge_attr,  # type: ignore
            map_from_feature=EDGE_RELATION,
        )

        print("Saving graph...")
        self.indexer.save(self.processed_paths[-1])

    def _retrieve_subgraphs(self) -> None:
        print("Encoding questions...")
        self.questions = [str(ds["question"]) for ds in self.raw_dataset]
        q_embs = self.model.encode(self.questions, batch_size=256)
        list_of_graphs = []
        print("Retrieving subgraphs...")
        textual_nodes = self.textual_nodes
        textual_edges = self.textual_edges
        if self.whole_graph_retrieval:
            graph = self.indexer.to_data(node_feature_name="x",
                                         edge_feature_name="edge_attr")
        else:
            graph_gen = get_features_for_triplets_groups(
                self.indexer, (ds['graph'] for ds in self.raw_dataset),
                pre_transform=preprocess_triplet)

        for index in tqdm(range(len(self.raw_dataset))):
            data_i = self.raw_dataset[index]
            if not self.whole_graph_retrieval:
                graph = next(graph_gen)
                textual_nodes = self.textual_nodes.iloc[
                    graph["node_idx"]].reset_index()
                textual_edges = self.textual_edges.iloc[
                    graph["edge_idx"]].reset_index()
            pcst_subgraph, desc = retrieval_via_pcst(
                graph,
                q_embs[index],
                textual_nodes,
                textual_edges,
                topk=3,
                topk_e=5,
                cost_e=0.5,
                override=self.override,
            )
            question = f"Question: {data_i['question']}\nAnswer: "
            label = ("|").join(data_i["answer"]).lower()

            pcst_subgraph["question"] = question
            pcst_subgraph["label"] = label
            pcst_subgraph["desc"] = desc
            list_of_graphs.append(pcst_subgraph.to("cpu"))
        print("Saving subgraphs...")
        self.save(list_of_graphs, self.processed_paths[0])

    def process(self) -> None:
        self._load_raw_data()
        self.model = SentenceTransformer(
            'sentence-transformers/all-roberta-large-v1').to(self.device)
        self.model.eval()
        if not os.path.exists(self.processed_paths[-1]):
            print("Encoding graph...")
            self._build_graph()
        else:
            print("Loading graph...")
            self.indexer = LargeGraphIndexer.from_disk(
                self.processed_paths[-1])
        self.textual_nodes = DataFrame.from_dict(
            {"node_attr": self.indexer.get_node_features()})
        self.textual_nodes["node_id"] = self.textual_nodes.index
        self.textual_nodes = self.textual_nodes[["node_id", "node_attr"]]
        self.textual_edges = DataFrame(self.indexer.get_edge_features(),
                                       columns=["src", "edge_attr", "dst"])
        self.textual_edges["src"] = [
            self.indexer._nodes[h] for h in self.textual_edges["src"]
        ]
        self.textual_edges["dst"] = [
            self.indexer._nodes[h] for h in self.textual_edges["dst"]
        ]
        self._retrieve_subgraphs()
