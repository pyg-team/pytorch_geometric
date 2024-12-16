# Code adapted from the G-Retriever paper: https://arxiv.org/abs/2402.07630
from typing import Any, Dict, List, Tuple, no_type_check

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.nn.nlp import SentenceTransformer


@no_type_check
def retrieval_via_pcst(
    data: Data,
    q_emb: Tensor,
    textual_nodes: Any,
    textual_edges: Any,
    topk: int = 3,
    topk_e: int = 3,
    cost_e: float = 0.5,
) -> Tuple[Data, str]:
    c = 0.01

    from pcst_fast import pcst_fast

    root = -1
    num_clusters = 1
    pruning = 'gw'
    verbosity_level = 0
    if topk > 0:
        n_prizes = torch.nn.CosineSimilarity(dim=-1)(q_emb, data.x)
        topk = min(topk, data.num_nodes)
        _, topk_n_indices = torch.topk(n_prizes, topk, largest=True)

        n_prizes = torch.zeros_like(n_prizes)
        n_prizes[topk_n_indices] = torch.arange(topk, 0, -1).float()
    else:
        n_prizes = torch.zeros(data.num_nodes)

    if topk_e > 0:
        e_prizes = torch.nn.CosineSimilarity(dim=-1)(q_emb, data.edge_attr)
        topk_e = min(topk_e, e_prizes.unique().size(0))

        topk_e_values, _ = torch.topk(e_prizes.unique(), topk_e, largest=True)
        e_prizes[e_prizes < topk_e_values[-1]] = 0.0
        last_topk_e_value = topk_e
        for k in range(topk_e):
            indices = e_prizes == topk_e_values[k]
            value = min((topk_e - k) / sum(indices), last_topk_e_value - c)
            e_prizes[indices] = value
            last_topk_e_value = value * (1 - c)
        # reduce the cost of the edges such that at least one edge is selected
        cost_e = min(cost_e, e_prizes.max().item() * (1 - c / 2))
    else:
        e_prizes = torch.zeros(data.num_edges)

    costs = []
    edges = []
    virtual_n_prizes = []
    virtual_edges = []
    virtual_costs = []
    mapping_n = {}
    mapping_e = {}
    for i, (src, dst) in enumerate(data.edge_index.t().numpy()):
        prize_e = e_prizes[i]
        if prize_e <= cost_e:
            mapping_e[len(edges)] = i
            edges.append((src, dst))
            costs.append(cost_e - prize_e)
        else:
            virtual_node_id = data.num_nodes + len(virtual_n_prizes)
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

    selected_nodes = vertices[vertices < data.num_nodes]
    selected_edges = [mapping_e[e] for e in edges if e < num_edges]
    virtual_vertices = vertices[vertices >= data.num_nodes]
    if len(virtual_vertices) > 0:
        virtual_vertices = vertices[vertices >= data.num_nodes]
        virtual_edges = [mapping_n[i] for i in virtual_vertices]
        selected_edges = np.array(selected_edges + virtual_edges)

    edge_index = data.edge_index[:, selected_edges]
    selected_nodes = np.unique(
        np.concatenate(
            [selected_nodes, edge_index[0].numpy(), edge_index[1].numpy()]))

    n = textual_nodes.iloc[selected_nodes]
    e = textual_edges.iloc[selected_edges]
    desc = n.to_csv(index=False) + '\n' + e.to_csv(
        index=False, columns=['src', 'edge_attr', 'dst'])

    mapping = {n: i for i, n in enumerate(selected_nodes.tolist())}
    src = [mapping[i] for i in edge_index[0].tolist()]
    dst = [mapping[i] for i in edge_index[1].tolist()]

    data = Data(
        x=data.x[selected_nodes],
        edge_index=torch.tensor([src, dst]),
        edge_attr=data.edge_attr[selected_edges],
    )

    return data, desc


class WebQSPDataset(InMemoryDataset):
    r"""The WebQuestionsSP dataset of the `"The Value of Semantic Parse
    Labeling for Knowledge Base Question Answering"
    <https://aclanthology.org/P16-2033/>`_ paper.

    Args:
        root (str): Root directory where the dataset should be saved.
        split (str, optional): If :obj:`"train"`, loads the training dataset.
            If :obj:`"val"`, loads the validation dataset.
            If :obj:`"test"`, loads the test dataset. (default: :obj:`"train"`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
        use_pcst (bool, optional): Whether to preprocess the dataset's graph
            with PCST or return the full graphs. (default: :obj:`True`)
    """
    def __init__(
        self,
        root: str,
        split: str = "train",
        force_reload: bool = False,
        use_pcst: bool = True,
    ) -> None:
        self.use_pcst = use_pcst
        super().__init__(root, force_reload=force_reload)

        if split not in {'train', 'val', 'test'}:
            raise ValueError(f"Invalid 'split' argument (got {split})")

        path = self.processed_paths[['train', 'val', 'test'].index(split)]
        self.load(path)

    @property
    def processed_file_names(self) -> List[str]:
        return ['train_data.pt', 'val_data.pt', 'test_data.pt']

    def process(self) -> None:
        import datasets
        import pandas as pd

        datasets = datasets.load_dataset('rmanluo/RoG-webqsp')

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_name = 'sentence-transformers/all-roberta-large-v1'
        model = SentenceTransformer(model_name).to(device)
        model.eval()

        for dataset, path in zip(
            [datasets['train'], datasets['validation'], datasets['test']],
                self.processed_paths,
        ):
            questions = [example["question"] for example in dataset]
            question_embs = model.encode(
                questions,
                batch_size=256,
                output_device='cpu',
            )

            data_list = []
            for i, example in enumerate(tqdm(dataset)):
                raw_nodes: Dict[str, int] = {}
                raw_edges = []
                for tri in example["graph"]:
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
                    "node_attr": k,
                } for k, v in raw_nodes.items()],
                                     columns=["node_id", "node_attr"])
                edges = pd.DataFrame(raw_edges,
                                     columns=["src", "edge_attr", "dst"])

                nodes.node_attr = nodes.node_attr.fillna("")
                x = model.encode(
                    nodes.node_attr.tolist(),
                    batch_size=256,
                    output_device='cpu',
                )
                edge_attr = model.encode(
                    edges.edge_attr.tolist(),
                    batch_size=256,
                    output_device='cpu',
                )
                edge_index = torch.tensor([
                    edges.src.tolist(),
                    edges.dst.tolist(),
                ], dtype=torch.long)

                question = f"Question: {example['question']}\nAnswer: "
                label = ('|').join(example['answer']).lower()
                data = Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                )
                if self.use_pcst and len(nodes) > 0 and len(edges) > 0:
                    data, desc = retrieval_via_pcst(
                        data,
                        question_embs[i],
                        nodes,
                        edges,
                        topk=3,
                        topk_e=5,
                        cost_e=0.5,
                    )
                else:
                    desc = nodes.to_csv(index=False) + "\n" + edges.to_csv(
                        index=False,
                        columns=["src", "edge_attr", "dst"],
                    )

                data.question = question
                data.label = label
                data.desc = desc
                data_list.append(data)

            self.save(data_list, path)
