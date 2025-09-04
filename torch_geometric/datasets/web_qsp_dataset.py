# Code adapted from the G-Retriever paper: https://arxiv.org/abs/2402.07630
import gc
import os
from itertools import chain
from typing import Any, Dict, Iterator, List, Optional

import torch
from tqdm import tqdm

from torch_geometric.data import InMemoryDataset
from torch_geometric.llm.large_graph_indexer import (
    EDGE_RELATION,
    LargeGraphIndexer,
    TripletLike,
    get_features_for_triplets_groups,
)
from torch_geometric.llm.models import SentenceTransformer
from torch_geometric.llm.utils.backend_utils import retrieval_via_pcst


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
        load_dataset_kwargs (dict, optional):
            Keyword arguments for the `datasets.load_dataset` function.
            (default: :obj:`{}`)
        retrieval_kwargs (dict, optional):
            Keyword arguments for the
            `get_features_for_triplets_groups` function.
            (default: :obj:`{}`)
    """
    def __init__(
        self,
        dataset_name: str,
        root: str,
        split: str = "train",
        force_reload: bool = False,
        verbose: bool = False,
        use_pcst: bool = True,
        load_dataset_kwargs: Optional[Dict[str, Any]] = None,
        retrieval_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.split = split
        self.dataset_name = dataset_name
        self.use_pcst = use_pcst
        self.load_dataset_kwargs = load_dataset_kwargs or {}
        """
        NOTE: If running into memory issues,
        try reducing this batch size for the LargeGraphIndexer
        used to build our KG.
        Example: self.retrieval_kwargs = {"batch_size": 64}
        """
        self.retrieval_kwargs = retrieval_kwargs or {}

        # Caching custom subsets of the dataset results in unsupported behavior
        if 'split' in self.load_dataset_kwargs:
            print("WARNING: Caching custom subsets of the dataset \
                results in unsupported behavior.\
                Please specify a separate root directory for each split,\
                or set force_reload=True on subsequent instantiations\
                of the dataset.")

        self.required_splits = ['train', 'validation', 'test']

        self.verbose = verbose
        self.force_reload = force_reload
        super().__init__(root, force_reload=force_reload)
        """
        NOTE: Current behavior is to process the entire dataset,
        and only return the split specified by the user.
        """
        if f'{split}_data.pt' not in set(self.processed_file_names):
            raise ValueError(f"Invalid 'split' argument (got {split})")
        if split == 'val':
            split = 'validation'

        self.load(self.processed_paths[self.required_splits.index(split)])

    @property
    def raw_file_names(self) -> List[str]:
        return ["raw.pt"]

    @property
    def processed_file_names(self) -> List[str]:
        return ["train_data.pt", "val_data.pt", "test_data.pt"]

    def download(self) -> None:
        import datasets

        # HF Load Dataset by dataset name if no path is specified
        self.load_dataset_kwargs['path'] = self.load_dataset_kwargs.get(
            'path', self.dataset_name)
        raw_dataset = datasets.load_dataset(**self.load_dataset_kwargs)

        # Assert that the dataset contains the required splits
        assert all(split in raw_dataset for split in self.required_splits), \
            f"Dataset '{self.dataset_name}' is missing required splits: \
            {self.required_splits}"

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
        x = self.model.encode(nodes, batch_size=256, output_device='cpu')
        self.indexer.add_node_feature(new_feature_name="x", new_feature_vals=x)

        # Edges:
        print("\tEncoding edges...")
        edges = self.indexer.get_unique_edge_features(
            feature_name=EDGE_RELATION)
        edge_attr = self.model.encode(edges, batch_size=256,
                                      output_device='cpu')
        self.indexer.add_edge_feature(
            new_feature_name="edge_attr",
            new_feature_vals=edge_attr,
            map_from_feature=EDGE_RELATION,
        )

        print("\tSaving graph...")
        self.indexer.save(self.indexer_path)

    def _retrieve_subgraphs(self) -> None:
        raw_splits = [
            self.raw_dataset[split] for split in self.required_splits
        ]
        zipped = zip(
            self.required_splits,
            raw_splits,  # noqa
            self.processed_paths,
        )
        for split_name, dataset, path in zipped:
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
        self.indexer_path = os.path.join(self.processed_dir,
                                         "large_graph_indexer")
        if self.force_reload or not os.path.exists(self.indexer_path):
            self._build_graph()
        else:
            print("Loading graph...")
            self.indexer = LargeGraphIndexer.from_disk(self.indexer_path)
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
        load_dataset_kwargs (dict, optional):
            Keyword arguments for the `datasets.load_dataset` function.
            (default: :obj:`{}`)
        retrieval_kwargs (dict, optional):
            Keyword arguments for the
            `get_features_for_triplets_groups` function.
            (default: :obj:`{}`)
    """
    def __init__(
        self,
        root: str,
        split: str = "train",
        force_reload: bool = False,
        verbose: bool = False,
        use_pcst: bool = True,
        load_dataset_kwargs: Optional[Dict[str, Any]] = None,
        retrieval_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        load_dataset_kwargs = load_dataset_kwargs or {}
        retrieval_kwargs = retrieval_kwargs or {}
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
        load_dataset_kwargs (dict, optional):
            Keyword arguments for the `datasets.load_dataset` function.
            (default: :obj:`{}`)
        retrieval_kwargs (dict, optional):
            Keyword arguments for the
            `get_features_for_triplets_groups` function.
            (default: :obj:`{}`)
    """
    def __init__(
        self,
        root: str,
        split: str = "train",
        force_reload: bool = False,
        verbose: bool = False,
        use_pcst: bool = True,
        load_dataset_kwargs: Optional[Dict[str, Any]] = None,
        retrieval_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        load_dataset_kwargs = load_dataset_kwargs or {}
        retrieval_kwargs = retrieval_kwargs or {}
        dataset_name = 'rmanluo/RoG-cwq'
        super().__init__(dataset_name, root, split, force_reload, verbose,
                         use_pcst, load_dataset_kwargs=load_dataset_kwargs,
                         retrieval_kwargs=retrieval_kwargs)
