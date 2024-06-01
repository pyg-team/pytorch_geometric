import os
from itertools import chain
from typing import Iterator, Optional

from large_graph_indexer import (
    EDGE_RELATION,
    LargeGraphIndexer,
    TripletLike,
    get_features_for_triplets,
)
from torch_geometric.datasets.web_qsp_dataset import *


def preprocess_triplet(triplet: TripletLike) -> TripletLike:
    h, r, t = triplet
    return h.lower(), r, t.lower()


class UpdatedWebQSPDataset(WebQSPDataset):

    def __init__(
        self,
        root: str = "",
        force_reload: bool = False,
        whole_graph_retrieval: bool = False,
        limit: int = -1,
    ) -> None:
        self.limit = limit
        self.whole_graph_retrieval = whole_graph_retrieval
        super().__init__(root, force_reload)

    @property
    def raw_file_names(self) -> List[str]:
        return ["raw_data", "split_idxs"]

    @property
    def processed_file_names(self) -> List[str]:
        return [
            "list_of_graphs.pt",
            "pre_filter.pt",
            "pre_transform.pt",
            "raw_graphs.pt",
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
        super().download()
        if self.limit >= 0:
            self.raw_dataset = self.raw_dataset.select(range(self.limit))
        self._save_raw_data()

    def _get_trips(self) -> Iterator[TripletLike]:
        return chain.from_iterable((iter(ds["graph"]) for ds in self.raw_dataset))

    def _build_graph(self) -> None:
        trips = self._get_trips()
        self.indexer: LargeGraphIndexer = LargeGraphIndexer.from_triplets(
            trips, pre_transform=preprocess_triplet
        )

        # Nodes:
        nodes = self.indexer.get_unique_node_features()
        x = self.model.encode(nodes, batch_size=256)
        self.indexer.add_node_feature(new_feature_name="x", new_feature_vals=x)

        # Edges:
        edges = self.indexer.get_unique_edge_features(feature_name=EDGE_RELATION)
        edge_attr = self.model.encode(edges, batch_size=256)
        self.indexer.add_edge_feature(
            new_feature_name="edge_attr",
            new_feature_vals=edge_attr,
            map_from_feature=EDGE_RELATION,
        )

        self.indexer.save(self.processed_paths[-1])

    def _retrieve_subgraphs(self) -> None:
        print("Encoding questions...")
        self.questions = [ds["question"] for ds in self.raw_dataset]
        q_embs = self.model.encode(self.questions, batch_size=256)
        list_of_graphs = []
        self.raw_graphs = []
        print("Retrieving subgraphs...")
        textual_nodes = self.textual_nodes
        textual_edges = self.textual_edges
        for index in tqdm(range(len(self.raw_dataset))):
            data_i = self.raw_dataset[index]
            local_trips = data_i["graph"]
            if self.whole_graph_retrieval:
                graph = self.indexer.to_data(
                    node_feature_name="x", edge_feature_name="edge_attr"
                )
            else:
                graph = get_features_for_triplets(
                    self.indexer, local_trips, pre_transform=preprocess_triplet
                )
                textual_nodes = self.textual_nodes.iloc[graph["node_idx"]].reset_index()
                textual_edges = self.textual_edges.iloc[graph["edge_idx"]].reset_index()
                self.raw_graphs.append(graph)
            pcst_subgraph, desc = retrieval_via_pcst(
                graph,
                q_embs[index],
                textual_nodes,
                textual_edges,
                topk=3,
                topk_e=5,
                cost_e=0.5,
            )
            question = f"Question: {data_i['question']}\nAnswer: "
            label = ("|").join(data_i["answer"]).lower()

            pcst_subgraph["question"] = question
            pcst_subgraph["label"] = label
            pcst_subgraph["desc"] = desc
            list_of_graphs.append(pcst_subgraph.to("cpu"))
        torch.save(self.raw_graphs, self.processed_paths[-2])
        self.save(list_of_graphs, self.processed_paths[0])

    def process(self) -> None:
        self._load_raw_data()
        self.model = SentenceTransformer().to(self.device)
        self.model.eval()
        if not os.path.exists(self.processed_dir[-1]):
            print("Encoding graph...")
            self._build_graph()
        else:
            print("Loading graph...")
            self.indexer = LargeGraphIndexer.from_disk(self.processed_dir[-1])
        self.textual_nodes = DataFrame.from_dict(
            {"node_attr": self.indexer.get_node_features()}
        )
        self.textual_nodes["node_id"] = self.textual_nodes.index
        self.textual_nodes = self.textual_nodes[["node_id", "node_attr"]]
        self.textual_edges = DataFrame(
            self.indexer.get_edge_features(), columns=["src", "edge_attr", "dst"]
        )
        self.textual_edges["src"] = [
            self.indexer._nodes[h] for h in self.textual_edges["src"]
        ]
        self.textual_edges["dst"] = [
            self.indexer._nodes[h] for h in self.textual_edges["dst"]
        ]
        self._retrieve_subgraphs()
