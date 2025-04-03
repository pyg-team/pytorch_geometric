# Code adapted from the G-Retriever paper: https://arxiv.org/abs/2402.07630
import gc
import os
from itertools import chain
from typing import Any, Dict, Iterator, List, Tuple, no_type_check

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
    topk_e: int = 5,
    cost_e: float = 0.5,
    num_clusters: int = 1,
) -> Tuple[Data, str]:

    # skip PCST for bad graphs
    booly = data.edge_attr is None or data.edge_attr.numel() == 0
    booly = booly or data.x is None or data.x.numel() == 0
    booly = booly or data.edge_index is None or data.edge_index.numel() == 0
    if not booly:
        c = 0.01

        from pcst_fast import pcst_fast

        root = -1
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

            topk_e_values, _ = torch.topk(e_prizes.unique(), topk_e,
                                          largest=True)
            e_prizes[e_prizes < topk_e_values[-1]] = 0.0
            last_topk_e_value = topk_e
            for k in range(topk_e):
                indices = e_prizes == topk_e_values[k]
                value = min((topk_e - k) / sum(indices), last_topk_e_value - c)
                e_prizes[indices] = value
                last_topk_e_value = value * (1 - c)
            # reduce the cost of the edges so that at least one edge is chosen
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
                [selected_nodes, edge_index[0].numpy(),
                 edge_index[1].numpy()]))

        n = textual_nodes.iloc[selected_nodes]
        e = textual_edges.iloc[selected_edges]
    else:
        n = textual_nodes
        e = textual_edges
    desc = n.to_csv(index=False) + '\n' + e.to_csv(
        index=False, columns=['src', 'edge_attr', 'dst'])

    mapping = {n: i for i, n in enumerate(selected_nodes.tolist())}
    src = [mapping[i] for i in edge_index[0].tolist()]
    dst = [mapping[i] for i in edge_index[1].tolist()]

    # HACK Added so that the subset of nodes and edges selected can be tracked
    node_idx = np.array(data.node_idx)[selected_nodes]
    edge_idx = np.array(data.edge_idx)[selected_edges]

    data = Data(
        x=data.x[selected_nodes],
        edge_index=torch.tensor([src, dst]).to(torch.long),
        edge_attr=data.edge_attr[selected_edges],
        # HACK Added so that the subset of nodes and edges selected can be tracked
        node_idx=node_idx,
        edge_idx=edge_idx,
    )

    return data, desc


def preprocess_triplet(triplet: TripletLike) -> TripletLike:
    h, r, t = triplet
    return str(h).lower(), str(r).lower(), str(t).lower()


class KGQABaseDataset(InMemoryDataset):
    r"""Base class for the 2 KGQA datasets used in `"Reasoning on Graphs:
    Faithful and Interpretable Large Language Model Reasoning"
    <https://arxiv.org/pdf/2310.01061>`_ paper.

    Args:
        dataset_name (str): HuggingFace `dataset` name.
        root (str): Root directory where the dataset should be saved.
        split (str, optional): If :obj:`"train"`, loads the training dataset.
            If :obj:`"val"`, loads the validation dataset.
            If :obj:`"test"`, loads the test dataset. (default: :obj:`"train"`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
        verbose (bool, optional): Whether to print output. Defaults to False.
        use_pcst (bool, optional): Whether to preprocess the dataset's graph
            with PCST or return the full graphs. (default: :obj:`True`)
        use_cwq (bool, optional): Whether to load the ComplexWebQuestions dataset. (default: :obj:`True`)
        load_dataset_kwargs (dict, optional): Keyword arguments for the `datasets.load_dataset` function. (default: :obj:`{}`)
        retrieval_kwargs (dict, optional): Keyword arguments for the the `get_features_for_triplets_groups` function. (default: :obj:`{}`)
    """
    def __init__(
            self,
            dataset_name: str,
            root: str,
            split: str = "train",
            force_reload: bool = False,
            verbose: bool = False,
            use_pcst: bool = True,
            use_cwq: bool = True,
            load_dataset_kwargs: Dict[str, Any] = dict(),
            retrieval_kwargs: Dict[str, Any] = dict(),
    ) -> None:
        self.split = split
        self.dataset_name = dataset_name
        self.use_pcst = use_pcst
        self.load_dataset_kwargs = load_dataset_kwargs

        # NOTE: If running into memory issues, try reducing this batch size
        self.retrieval_kwargs = retrieval_kwargs

        # Caching custom subsets of the dataset results in unsupported behavior
        if 'split' in load_dataset_kwargs:
            print(
                "WARNING: Caching custom subsets of the dataset results in unsupported behavior. Please specify a separate root directory for each split, or set force_reload=True on subsequent instantiations of the dataset."
            )

        self.required_splits = ['train', 'validation', 'test']

        self.verbose = verbose
        self.force_reload = force_reload
        super().__init__(root, force_reload=force_reload)

        # NOTE: Current behavior is to process the entire dataset, and only return the split specified by the user
        if f'{split}_data.pt' not in set(self.processed_file_names):
            raise ValueError(f"Invalid 'split' argument (got {split})")
        if split == 'val':
            split = 'validation'

        self.load(self.processed_paths[self.required_splits.index(split)])

    @property
    def raw_file_names(self) -> List[str]:
        return ["raw.pt"]

    @property
    def _processed_split_file_names(self) -> List[str]:
        return ["train_data.pt", "val_data.pt", "test_data.pt"]

    @property
    def processed_file_names(self) -> List[str]:
        return self._processed_split_file_names + ["large_graph_indexer"]

    def download(self) -> None:
        import datasets

        # HF Load Dataset by dataset name if no path is specified
        self.load_dataset_kwargs['path'] = self.load_dataset_kwargs.get(
            'path', self.dataset_name)
        raw_dataset = datasets.load_dataset(**self.load_dataset_kwargs)

        # Assert that the dataset contains the required splits
        assert all(split in raw_dataset for split in self.required_splits), \
            f"Dataset '{self.dataset_name}' is missing required splits: {self.required_splits}"

        raw_dataset.save_to_disk(self.raw_paths[0])

    def _get_trips(self) -> Iterator[TripletLike]:
        # Iterate over each element's graph in each split of the dataset
        # Using chain to lazily iterate without storing all trips in memory
        split_iterators = []

        for split in self.required_splits:
            # Create an iterator for each element's graph in the current split
            split_graphs = (element['graph']
                            for element in self.raw_dataset[split])
            split_iterators.append(chain.from_iterable(split_graphs))

        # Chain all split iterators together
        return chain.from_iterable(split_iterators)

    def _build_graph(self) -> None:
        print("Encoding graph...")
        trips = self._get_trips()
        self.indexer: LargeGraphIndexer = LargeGraphIndexer.from_triplets(
            trips, pre_transform=preprocess_triplet)

        # Nodes:
        print("\tEncoding nodes...")
        nodes = self.indexer.get_unique_node_features()
        x = self.model.encode(
            nodes,  # type: ignore
            batch_size=256,
            output_device='cpu')
        self.indexer.add_node_feature(new_feature_name="x", new_feature_vals=x)

        # Edges:
        print("\tEncoding edges...")
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

        print("\tSaving graph...")
        self.indexer.save(self.processed_paths[-1])

    def _retrieve_subgraphs(self) -> None:
        for split_name, dataset, path in zip(
                self.required_splits,
            [self.raw_dataset[split] for split in self.required_splits],
                self.processed_paths,
        ):
            print(f"Processing {split_name} split...")

            print("\tEncoding questions...")
            split_questions = [str(element['question']) for element in dataset]
            split_q_embs = self.model.encode(split_questions, batch_size=256,
                                             output_device='cpu')

            print("\tRetrieving subgraphs...")
            results_graphs = []
            retrieval_kwargs = {
                **self.retrieval_kwargs,
                **{
                    'pre_transform': preprocess_triplet,
                    'verbose': self.verbose,
                }
            }
            graph_gen = get_features_for_triplets_groups(
                self.indexer, (element['graph'] for element in dataset),
                **retrieval_kwargs)

            for index in tqdm(range(len(dataset)), disable=not self.verbose):
                data_i = dataset[index]
                graph = next(graph_gen)
                textual_nodes = self.textual_nodes.iloc[
                    graph["node_idx"]].reset_index()
                textual_edges = self.textual_edges.iloc[
                    graph["edge_idx"]].reset_index()
                if self.use_pcst and len(textual_nodes) > 0 and len(
                        textual_edges) > 0:
                    subgraph, desc = retrieval_via_pcst(
                        graph,
                        split_q_embs[index],
                        textual_nodes,
                        textual_edges,
                    )
                else:
                    desc = textual_nodes.to_csv(
                        index=False) + "\n" + textual_edges.to_csv(
                            index=False,
                            columns=["src", "edge_attr", "dst"],
                        )
                    subgraph = graph
                question = f"Question: {data_i['question']}\nAnswer: "
                label = ("|").join(data_i["answer"]).lower()

                subgraph["question"] = question
                subgraph["label"] = label
                subgraph["desc"] = desc
                results_graphs.append(subgraph.to("cpu"))
            print("\tSaving subgraphs...")
            self.save(results_graphs, path)

    def process(self) -> None:
        import datasets
        from pandas import DataFrame
        self.raw_dataset = datasets.load_from_disk(self.raw_paths[0])

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_name = 'sentence-transformers/all-roberta-large-v1'
        self.model: SentenceTransformer = SentenceTransformer(model_name).to(
            device)
        self.model.eval()
        if self.force_reload or not os.path.exists(self.processed_paths[-1]):
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

        gc.collect()
        torch.cuda.empty_cache()


class WebQSPDataset(KGQABaseDataset):
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
        verbose (bool, optional): Whether to print output. Defaults to False.
        use_pcst (bool, optional): Whether to preprocess the dataset's graph
            with PCST or return the full graphs. (default: :obj:`True`)
        load_dataset_kwargs (dict, optional): Keyword arguments for the `datasets.load_dataset` function. (default: :obj:`{}`)
        retrieval_kwargs (dict, optional): Keyword arguments for the `get_features_for_triplets_groups` function. (default: :obj:`{}`)
    """
    def __init__(
        self, root: str, split: str = "train", force_reload: bool = False,
        verbose: bool = False, use_pcst: bool = True,
        load_dataset_kwargs: Dict[str, Any] = dict(),
        retrieval_kwargs: Dict[str, Any] = dict()
    ) -> None:
        # Modify these paramters if running into memory/compute issues
        default_retrieval_kwargs = {
            'max_batch_size': 250,  # Lower batch size to reduce memory usage
            'num_workers':
            None,  # Use all available workers, or set to number of threads
        }
        retrieval_kwargs = {**default_retrieval_kwargs, **retrieval_kwargs}
        dataset_name = 'rmanluo/RoG-webqsp'
        super().__init__(dataset_name, root, split, force_reload, verbose,
                         use_pcst, load_dataset_kwargs=load_dataset_kwargs,
                         retrieval_kwargs=retrieval_kwargs)


class CWQDataset(KGQABaseDataset):
    r"""The ComplexWebQuestions (CWQ) dataset of the `"The Web as a
    Knowledge-base forAnswering Complex Questions"
    <https://arxiv.org/pdf/1803.06643>`_ paper.

    Args:
        root (str): Root directory where the dataset should be saved.
        split (str, optional): If :obj:`"train"`, loads the training dataset.
            If :obj:`"val"`, loads the validation dataset.
            If :obj:`"test"`, loads the test dataset. (default: :obj:`"train"`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
        verbose (bool, optional): Whether to print output. Defaults to False.
        use_pcst (bool, optional): Whether to preprocess the dataset's graph
            with PCST or return the full graphs. (default: :obj:`True`)
        load_dataset_kwargs (dict, optional): Keyword arguments for the `datasets.load_dataset` function. (default: :obj:`{}`)
        retrieval_kwargs (dict, optional): Keyword arguments for the `get_features_for_triplets_groups` function. (default: :obj:`{}`)
    """
    def __init__(
        self, root: str, split: str = "train", force_reload: bool = False,
        verbose: bool = False, use_pcst: bool = True,
        load_dataset_kwargs: Dict[str, Any] = dict(),
        retrieval_kwargs: Dict[str, Any] = dict()
    ) -> None:
        dataset_name = 'rmanluo/RoG-cwq'
        super().__init__(dataset_name, root, split, force_reload, verbose,
                         use_pcst, load_dataset_kwargs=load_dataset_kwargs,
                         retrieval_kwargs=retrieval_kwargs)
