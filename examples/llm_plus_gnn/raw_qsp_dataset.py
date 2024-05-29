from torch_geometric.datasets.web_qsp_dataset import *
from typing import Optional


class RawWebQSPDataset(WebQSPDataset):

    def __init__(
        self,
        root: str = "",
        force_reload: bool = False,
        with_process: bool = False,
        limit: Optional[int] = None,
    ) -> None:
        self.with_process = with_process
        self.limit = limit
        if self.with_process:
            super().__init__(root, force_reload)
        else:
            self._check_dependencies()
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            super(InMemoryDataset, self).__init__(
                root, None, None, force_reload=force_reload
            )
            self._load_raw_data()

    @property
    def raw_file_names(self) -> List[str]:
        if not self.with_process:
            return ["raw_data", "split_idxs"]
        else:
            return []

    def _save_raw_data(self) -> None:
        self.raw_dataset.save_to_disk(self.raw_paths[0])
        torch.save(self.split_idxs, self.raw_paths[1])

    def _load_raw_data(self) -> None:
        self.raw_dataset = datasets.load_from_disk(self.raw_paths[0])
        self.split_idxs = torch.load(self.raw_paths[1])

    def download(self) -> None:
        super().download()
        if not self.with_process:
            self._save_raw_data()

    def process(self) -> None:
        if self.with_process:
            pretrained_repo = "sentence-transformers/all-roberta-large-v1"
            self.model = SentenceTransformer(pretrained_repo)
            self.model.to(self.device)
            self.model.eval()
            # self.questions = [i["question"] for i in self.raw_dataset]
            list_of_graphs = []
            # encode questions
            # print("Encoding questions...")
            # q_embs = text2embedding(self.model, self.device, self.questions)
            print("Encoding graphs...")
            limit = self.limit if self.limit else len(self.raw_dataset)
            for index in tqdm(range(limit)):
                data_i = self.raw_dataset[index]
                raw_nodes: Dict[str, int] = {}
                raw_edges = []
                for tri in data_i["graph"]:
                    h, r, t = tri
                    h = h.lower()
                    t = t.lower()
                    if h not in raw_nodes:
                        raw_nodes[h] = len(raw_nodes)
                    if t not in raw_nodes:
                        raw_nodes[t] = len(raw_nodes)
                    raw_edges.append(
                        {"src": raw_nodes[h], "edge_attr": r, "dst": raw_nodes[t]}
                    )
                nodes = pd.DataFrame(
                    [{"node_id": v, "node_attr": k} for k, v in raw_nodes.items()],
                    columns=["node_id", "node_attr"],
                )
                edges = pd.DataFrame(raw_edges, columns=["src", "edge_attr", "dst"])
                # encode nodes
                nodes.node_attr = nodes.node_attr.fillna("")
                x = text2embedding(self.model, self.device, nodes.node_attr.tolist())
                # encode edges
                edge_attr = text2embedding(
                    self.model, self.device, edges.edge_attr.tolist()
                )
                edge_index = torch.LongTensor([edges.src.tolist(), edges.dst.tolist()])
                question = f"Question: {data_i['question']}\nAnswer: "
                label = ("|").join(data_i["answer"]).lower()
                raw_graph = Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    num_nodes=len(nodes),
                ).to("cpu")
                list_of_graphs.append(raw_graph)
                # psct_subgraph, desc = retrieval_via_pcst(raw_graph, q_embs[index],
                #                                        nodes, edges, topk=3,
                #                                        topk_e=5, cost_e=0.5)
                # psct_subgraph["question"] = question
                # psct_subgraph["label"] = label
                # psct_subgraph["desc"] = desc
                # list_of_graphs.append(psct_subgraph.to("cpu"))
            self.save(list_of_graphs, self.processed_paths[0])
