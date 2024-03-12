from typing import List, Optional, Tuple

import numpy as np

try:
    import pandas as pd
    from pandas import DataFrame as df
    WITH_PANDAS = True
except ImportError as e:  # noqa
    df = None
    WITH_PANDAS = False
import torch
import torch.nn.functional as F

try:
    from pcst_fast import pcst_fast
    WITH_PCST = True
except ImportError as e:  # noqa
    WITH_PCST = False
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from transformers import AutoModel, AutoTokenizer
    WITH_TRANSFORMERS = True
except ImportError as e:  # noqa
    WITH_TRANSFORMERS = False
try:
    import datasets
    WITH_DATASETS = True
except ImportError as e:  # noqa
    WITH_DATASETS = False

from torch_geometric.data import Data, InMemoryDataset


class Dataset(torch.utils.data.Dataset):
    def __init__(self, input_ids: Optional[torch.Tensor] = None,
                 attention_mask: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.data = {
            "input_ids": input_ids,
            "att_mask": attention_mask,
        }

    def __len__(self):
        return self.data["input_ids"].size(0)

    def __getitem__(self, index: int):
        if isinstance(index, torch.Tensor):
            index = index.item()
        batch_data = dict()
        for key in self.data.keys():
            if self.data[key] is not None:
                batch_data[key] = self.data[key][index]
        return batch_data


class Sentence_Transformer(torch.nn.Module):
    def __init__(self, pretrained_repo: str) -> None:
        super(Sentence_Transformer, self).__init__()
        print(f"inherit model weights from {pretrained_repo}")
        self.bert_model = AutoModel.from_pretrained(pretrained_repo)

    def mean_pooling(self, token_embeddings: torch.Tensor,
                     attention_mask: torch.Tensor):
        data_type = token_embeddings.dtype
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()).to(data_type)
        return torch.sum(token_embeddings * input_mask_expanded,
                         1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, input_ids: torch.Tensor, att_mask: torch.Tensor):
        bert_out = self.bert_model(input_ids=input_ids,
                                   attention_mask=att_mask)

        # First element of model_output contains all token embeddings
        token_embeddings = bert_out[0]
        sentence_embeddings = self.mean_pooling(token_embeddings, att_mask)
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings


def sbert_text2embedding(model: Sentence_Transformer,
                         tokenizer: torch.nn.Module, device: torch.device,
                         text: List[str]) -> torch.Tensor:
    try:
        encoding = tokenizer(text, padding=True, truncation=True,
                             return_tensors="pt")
        dataset = Dataset(input_ids=encoding.input_ids,
                          attention_mask=encoding.attention_mask)

        # DataLoader
        dataloader = DataLoader(dataset, batch_size=256, shuffle=False)

        # Placeholder for storing the embeddings
        all_embeddings_list = []

        # Iterate through batches
        with torch.no_grad():

            for batch in dataloader:
                # Move batch to the appropriate device
                batch = {key: value.to(device) for key, value in batch.items()}

                # Forward pass
                embeddings = model(input_ids=batch["input_ids"],
                                   att_mask=batch["att_mask"])

                # Append the embeddings to the list
                all_embeddings_list.append(embeddings)

        # Concatenate the embeddings from all batches
        all_embeddings = torch.cat(all_embeddings_list, dim=0).cpu()
    except:  # noqa
        print(
            "SBERT text embedding failed, returning torch.zeros((0, 1024))...")
        return torch.zeros((0, 1024))

    return all_embeddings


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
        missing_imports = False
        missing_str = []
        if not WITH_PCST:
            missing_str.append('pcst_fast')
            missing_imports = True
        if not WITH_TRANSFORMERS:
            missing_str.append('transformers')
            missing_imports = True
        if not WITH_DATASETS:
            missing_str.append('datasets')
            missing_imports = True
        if not WITH_PANDAS:
            missing_str.append('pandas')
            missing_imports = True
        if missing_imports:
            missing_str = missing_str.join(' ')
            error_out = f"`pip install {missing_str}` to use this dataset."
            raise ImportError(error_out)
        self.prompt = "Please answer the given question."
        self.graph = None
        self.graph_type = "Knowledge Graph"
        self.model_name = "sbert"
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(root, None, None, force_reload=force_reload)
        self.load(self.processed_paths[0])

    def retrieval_via_pcst(self, graph: Data, q_emb: torch.Tensor,
                           textual_nodes: df, textual_edges: df, topk: int = 3,
                           topk_e: int = 3,
                           cost_e: float = 0.5) -> Tuple[Data, str]:
        # from original G-Retriever work
        # https://arxiv.org/abs/2402.07630
        c = 0.01
        if len(textual_nodes) == 0 or len(textual_edges) == 0:
            desc = textual_nodes.to_csv(
                index=False) + "\n" + textual_edges.to_csv(
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
            e_prizes = torch.nn.CosineSimilarity(dim=-1)(q_emb,
                                                         graph.edge_attr)
            topk_e = min(topk_e, e_prizes.unique().size(0))

            topk_e_values, _ = torch.topk(e_prizes.unique(), topk_e,
                                          largest=True)
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
        vritual_n_prizes = []
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
                virtual_node_id = graph.num_nodes + len(vritual_n_prizes)
                mapping_n[virtual_node_id] = i
                virtual_edges.append((src, virtual_node_id))
                virtual_edges.append((virtual_node_id, dst))
                virtual_costs.append(0)
                virtual_costs.append(0)
                vritual_n_prizes.append(prize_e - cost_e)

        prizes = np.concatenate([n_prizes, np.array(vritual_n_prizes)])
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
                [selected_nodes, edge_index[0].numpy(),
                 edge_index[1].numpy()]))

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
        self.model = Sentence_Transformer(pretrained_repo)
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_repo)
        self.text2embedding = sbert_text2embedding
        self.questions = [i["question"] for i in self.raw_dataset]
        list_of_graphs = []
        # encode questions
        print("Encoding questions...")
        q_embs = self.text2embedding(self.model, self.tokenizer, self.device,
                                     self.questions)
        print("Encoding graphs...")
        for index in tqdm(range(len(self.raw_dataset))):
            if index == 1:
                quit()
            data_i = self.raw_dataset[index]
            raw_nodes = {}
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
            nodes = pd.DataFrame([{
                "node_id": v,
                "node_attr": k
            } for k, v in raw_nodes.items()], columns=["node_id", "node_attr"])
            edges = pd.DataFrame(raw_edges,
                                 columns=["src", "edge_attr", "dst"])
            # encode nodes
            nodes.node_attr.fillna("", inplace=True)
            x = self.text2embedding(self.model, self.tokenizer, self.device,
                                    nodes.node_attr.tolist())
            # encode edges
            edge_attr = self.text2embedding(self.model, self.tokenizer,
                                            self.device,
                                            edges.edge_attr.tolist())
            edge_index = torch.LongTensor(
                [edges.src.tolist(), edges.dst.tolist()])
            question = f"Question: {data_i['question']}\nAnswer: "
            label = ("|").join(data_i["answer"]).lower()
            raw_graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                             num_nodes=len(nodes)).to("cpu")
            psct_subgraph, desc = self.retrieval_via_pcst(
                raw_graph, q_embs[index], nodes, edges, topk=3, topk_e=5,
                cost_e=0.5)
            psct_subgraph["question"] = question
            psct_subgraph["label"] = label
            psct_subgraph["desc"] = desc
            list_of_graphs.append(psct_subgraph.to("cpu"))
        self.save(list_of_graphs, self.processed_paths[0])
