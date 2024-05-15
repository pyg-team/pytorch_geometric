# Alot of code in this file is based on the original G-Retriever paper
# url: https://arxiv.org/abs/2402.07630
from typing import Dict, List, Tuple, no_type_check

import numpy as np
import torch
from tqdm import tqdm

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.nn.text import SentenceTransformer, text2embedding

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
) -> Tuple[Data, str]:
    c = 0.01
    if len(textual_nodes) == 0 or len(textual_edges) == 0:
        desc = textual_nodes.to_csv(index=False) + "\n" + textual_edges.to_csv(
            index=False, columns=["src", "edge_attr", "dst"])
        graph = Data(x=graph.x, edge_index=graph.edge_index,
                     edge_attr=graph.edge_attr, num_nodes=graph.num_nodes)
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
    desc = n.to_csv(index=False) + "\n" + e.to_csv(
        index=False, columns=["src", "edge_attr", "dst"])

    mapping = {n: i for i, n in enumerate(selected_nodes.tolist())}

    x = graph.x[selected_nodes]
    edge_attr = graph.edge_attr[selected_edges]
    src = [mapping[i] for i in edge_index[0].tolist()]
    dst = [mapping[i] for i in edge_index[1].tolist()]
    edge_index = torch.LongTensor([src, dst])
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                num_nodes=len(selected_nodes))

    return data, desc


class WebQSPDataset(InMemoryDataset):
    r"""The WebQuestionsSP dataset was released as part of
    “The Value of Semantic Parse Labeling for Knowledge
    Base Question Answering”
    [Yih, Richardson, Meek, Chang & Suh, 2016].
    It contains semantic parses, vs. answers, for a set of questions
    that originally comes from WebQuestions [Berant et al., 2013]."
    Processing based on "G-Retriever: Retrieval-Augmented Generation
    for Textual Graph Understanding and Question Answering".
    Requires datasets and transformers from HuggingFace.

    Args:
        root (str): Root directory where the dataset should be saved.
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
    """
    def __init__(
        self,
        root: str = "",
        force_reload: bool = False,
    ) -> None:
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
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(root, None, None, force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return []

    @property
    def processed_file_names(self) -> List[str]:
        return ["list_of_graphs.pt", "pre_filter.pt", "pre_transform.pt"]

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
            len(dataset["validation"])
        }

    def process(self) -> None:
        pretrained_repo = "sentence-transformers/all-roberta-large-v1"
        self.model = SentenceTransformer(pretrained_repo, device=self.device)
        self.model.eval()
        self.questions = [i["question"] for i in self.raw_dataset]
        list_of_graphs = []
        # encode questions
        print("Encoding questions...")
        q_embs = text2embedding(self.model, self.questions, device=self.device)
        print("Encoding graphs...")
        for index in tqdm(range(len(self.raw_dataset))):
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
                raw_edges.append({
                    "src": raw_nodes[h],
                    "edge_attr": r,
                    "dst": raw_nodes[t]
                })
            nodes = DataFrame([{
                "node_id": v,
                "node_attr": k
            } for k, v in raw_nodes.items()], columns=["node_id", "node_attr"])
            edges = DataFrame(raw_edges, columns=["src", "edge_attr", "dst"])
            # encode nodes
            nodes.node_attr = nodes.node_attr.fillna("")
            x = text2embedding(self.model,
                               nodes.node_attr.tolist(),
                               device=self.device)
            # encode edges
            edge_attr = text2embedding(self.model,
                                       edges.edge_attr.tolist(),
                                       device=self.device)
            edge_index = torch.LongTensor(
                [edges.src.tolist(), edges.dst.tolist()])
            question = f"Question: {data_i['question']}\nAnswer: "
            label = ("|").join(data_i["answer"]).lower()
            raw_graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                             num_nodes=len(nodes)).to("cpu")
            psct_subgraph, desc = retrieval_via_pcst(raw_graph, q_embs[index],
                                                     nodes, edges, topk=3,
                                                     topk_e=5, cost_e=0.5)
            psct_subgraph["question"] = question
            psct_subgraph["label"] = label
            psct_subgraph["desc"] = desc
            list_of_graphs.append(psct_subgraph.to("cpu"))
        self.save(list_of_graphs, self.processed_paths[0])
