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
    return tuple([s.lower() for s in triplet])


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
            "large_graph.pt",
        ]

    def _save_raw_data(self) -> None:
        self.raw_dataset.save_to_disk(self.raw_paths[0])
        torch.save(self.split_idxs, self.raw_paths[1])

    def _load_raw_data(self) -> None:
        self.raw_dataset = datasets.load_from_disk(self.raw_paths[0])
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
        x = text2embedding(self.model, self.device, nodes)
        self.indexer.add_node_feature(new_feature_name="x", new_feature_vals=x)

        # Edges:
        edges = self.indexer.get_unique_edge_features(feature_name=EDGE_RELATION)
        edge_attr = text2embedding(self.model, self.device, edges)
        self.indexer.add_edge_feature(
            new_feature_name="edge_attr",
            new_feature_vals=edge_attr,
            map_from_feature=EDGE_RELATION,
        )

        self.graph = self.indexer.to_data(
            node_feature_name="x", edge_feature_name="edge_attr"
        )
        self.save(self.graph, self.processed_paths[3])

    def _retrieve_subgraphs(self) -> None:
        print("Encoding questions...")
        q_embs = text2embedding(self.model, self.device, self.questions)
        list_of_graphs = []
        print("Retrieving subgraphs...")
        for index in tqdm(range(len(self.raw_dataset))):
            data_i = self.raw_dataset[index]
            local_trips = data_i["graph"]
            if self.whole_graph_retrieval:
                graph = self.graph
            else:
                graph = get_features_for_triplets(
                    self.indexer, local_trips, pre_transform=preprocess_triplet
                )
            pcst_subgraph, desc = retrieval_via_pcst(
                graph,
                q_embs[index],
                self.textual_nodes,
                self.textual_edges,
                topk=3,
                topk_e=5,
                cost_e=0.5,
            )
            question = f"Question: {data_i['question']}\nAnswer: "
            label = ("|").join(data_i["answer"]).lower()

            pcst_subgraph["question"] = question
            pcst_subgraph["label"] = label
            pcst_subgraph["desc"] - desc
            list_of_graphs.append(pcst_subgraph.to("cpu"))
        self.save(list_of_graphs, self.processed_paths[0])

    def process(self) -> None:
        pretrained_repo = "sentence-transformers/all-roberta-large-v1"
        self.model = SentenceTransformer(pretrained_repo)
        self.model.to(self.device)
        self.model.eval()
        self.questions = [ds["question"] for ds in self.raw_dataset]
        print("Encoding graph...")
        self._build_graph()
        self.textual_nodes = pd.DataFrame(
            {"node_attr": self.indexer.get_edge_features()}
        )
        self.textual_nodes["node_id"] = self.textual_nodes.index
        self.textual_edges = pd.DataFrame(
            self.indexer.get_edge_features, columns=["src", "edge_attr", "dst"]
        )
        self._retrieve_subgraphs()
