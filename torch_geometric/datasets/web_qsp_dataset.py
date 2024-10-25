# Code adapted from the G-Retriever paper: https://arxiv.org/abs/2402.07630
import os
from itertools import chain
from typing import Any, Iterator, List, Tuple, no_type_check

import numpy as np
import torch
from torch import Tensor
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


@no_type_check
def retrieval_via_pcst(
    data: Data,
    q_emb: Tensor,
    textual_nodes: Any,
    textual_edges: Any,
    topk: int = 3,
    topk_e: int = 3,
    cost_e: float = 0.5,
    save_idx: bool = False,
    override: bool = False,
) -> Tuple[Data, str]:
    c = 0.01
    if len(textual_nodes) == 0 or len(textual_edges) == 0 or override:
        desc = textual_nodes.to_csv(index=False) + "\n" + textual_edges.to_csv(
            index=False,
            columns=["src", "edge_attr", "dst"],
        )
        return data, desc

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

    # HACK Added so that the subset of nodes and edges selected can be tracked
    if save_idx:
        node_idx = np.array(data.node_idx)[selected_nodes]
        edge_idx = np.array(data.edge_idx)[selected_edges]

    data = Data(
        x=data.x[selected_nodes],
        edge_index=torch.tensor([src, dst]),
        edge_attr=data.edge_attr[selected_edges],
    )
    if save_idx:
        data['node_idx'] = node_idx
        data['edge_idx'] = edge_idx

    return data, desc


def preprocess_triplet(triplet: TripletLike) -> TripletLike:
    h, r, t = triplet
    return str(h).lower(), str(r), str(t).lower()


class WebQSPDataset(InMemoryDataset):
    r"""The WebQuestionsSP dataset of the `"The Value of Semantic Parse
    Labeling for Knowledge Base Question Answering"
    <https://aclanthology.org/P16-2033/>`_ paper.

    Args:
        root (str): Root directory where the dataset should be saved.
        split (str, optional): If :obj:`"train"`, loads the training dataset.
            If :obj:`"validation"`, loads the validation dataset.
            If :obj:`"test"`, loads the test dataset. (default: :obj:`"train"`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
        limit (int, optional): Construct only the first n samples.
            Defaults to -1 to construct all samples.
        include_pcst (bool, optional): Whether to include PCST step
            (See GRetriever paper). Defaults to True.
        verbose (bool, optional): Whether to print output. Defaults to False.
    """
    def __init__(
        self,
        root: str,
        split: str = "train",
        force_reload: bool = False,
        limit: int = -1,
        include_pcst: bool = True,
        verbose: bool = False,
    ) -> None:
        self.limit = limit
        self.split = split
        self.include_pcst = include_pcst
        # TODO Confirm why the dependency checks and device setting were removed here # noqa
        '''
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self._check_dependencies()
        '''
        self.verbose = verbose
        self.force_reload = force_reload
        super().__init__(root, force_reload=force_reload)

        if split not in set(self.raw_file_names):
            raise ValueError(f"Invalid 'split' argument (got {split})")

        self._load_raw_data()
        self.load(self.processed_paths[0] + "" * (self.limit >= 0))

    '''
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
    '''

    @property
    def raw_file_names(self) -> List[str]:
        return ["train", "validation", "test"]

    @property
    def processed_file_names(self) -> List[str]:
        file_lst = [
            "train_data.pt",
            "validation_data.pt",
            "test_data.pt",
            "pre_filter.pt",
            "pre_transform.pt",
            "large_graph_indexer",
        ]
        split_file = file_lst.pop(self.raw_file_names.index(self.split))
        file_lst.insert(0, split_file)
        return file_lst

    def _save_raw_data(self, dataset) -> None:
        for i, split in enumerate(self.raw_file_names):
            dataset[split].save_to_disk(self.raw_paths[i])

    def _load_raw_data(self) -> None:
        import datasets
        if not hasattr(self, "raw_dataset"):
            self.raw_dataset = datasets.load_from_disk(
                self.raw_paths[self.raw_file_names.index[self.split]])

        if self.limit >= 0:
            self.raw_dataset = self.raw_dataset.select(
                range(min(self.limit, len(self.raw_dataset))))

    def download(self) -> None:
        import datasets

        dataset = datasets.load_dataset("rmanluo/RoG-webqsp")
        self._save_raw_data(dataset)
        self.raw_dataset = dataset[self.split]

    def _get_trips(self) -> Iterator[TripletLike]:
        return chain.from_iterable(
            iter(ds["graph"]) for ds in self.raw_dataset)

    def _build_graph(self) -> None:
        trips = self._get_trips()
        self.indexer: LargeGraphIndexer = LargeGraphIndexer.from_triplets(
            trips, pre_transform=preprocess_triplet)

        # Nodes:
        nodes = self.indexer.get_unique_node_features()
        x = self.model.encode(
            nodes,  # type: ignore
            batch_size=256,
            output_device='cpu')
        self.indexer.add_node_feature(new_feature_name="x", new_feature_vals=x)

        # Edges:
        edges = self.indexer.get_unique_edge_features(
            feature_name=EDGE_RELATION)
        edge_attr = self.model.encode(
            edges,  # type: ignore
            batch_size=256,
            output_device='cpu')
        self.indexer.add_edge_feature(
            new_feature_name="edge_attr",
            new_feature_vals=edge_attr,
            map_from_feature=EDGE_RELATION,
        )

        print("Saving graph...")
        self.indexer.save(self.processed_paths[-1])

    def _retrieve_subgraphs(self) -> None:
        print("Encoding questions...")
        self.questions = [str(ds["question"]) for ds in self.raw_dataset]
        q_embs = self.model.encode(self.questions, batch_size=256,
                                   output_device='cpu')
        list_of_graphs = []
        print("Retrieving subgraphs...")
        textual_nodes = self.textual_nodes
        textual_edges = self.textual_edges
        graph_gen = get_features_for_triplets_groups(
            self.indexer, (ds['graph'] for ds in self.raw_dataset),
            pre_transform=preprocess_triplet, verbose=self.verbose)

        for index in tqdm(range(len(self.raw_dataset)),
                          disable=not self.verbose):
            data_i = self.raw_dataset[index]
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
                override=not self.include_pcst,
            )
            question = f"Question: {data_i['question']}\nAnswer: "
            label = ("|").join(data_i["answer"]).lower()

            pcst_subgraph["question"] = question
            pcst_subgraph["label"] = label
            pcst_subgraph["desc"] = desc
            list_of_graphs.append(pcst_subgraph.to("cpu"))
        print("Saving subgraphs...")
        self.save(list_of_graphs,
                  self.processed_paths[0] + "" * (self.limit >= 0))

    def process(self) -> None:
        from pandas import DataFrame
        self._load_raw_data()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(
            'sentence-transformers/all-roberta-large-v1').to(device)
        self.model.eval()
        if self.force_reload or not os.path.exists(self.processed_paths[-1]):
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
