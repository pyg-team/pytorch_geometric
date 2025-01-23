import gc
from collections.abc import Iterable, Iterator
from typing import Any, Dict, Optional, Type, Union

import torch
from torch import Tensor
from torch.nn import Module
from torchmetrics.functional import pairwise_cosine_similarity

from torch_geometric.data import Data, HeteroData
from torch_geometric.distributed import LocalFeatureStore
from torch_geometric.nn.nlp import SentenceTransformer
from torch_geometric.nn.pool import ApproxMIPSKNNIndex
from torch_geometric.sampler import HeteroSamplerOutput, SamplerOutput
from torch_geometric.typing import InputEdges, InputNodes


# NOTE: Only compatible with Homogeneous graphs for now
class KNNRAGFeatureStore(LocalFeatureStore):
    """A feature store that uses a KNN-based approach to retrieve seed nodes and edges.
    """
    def __init__(self, enc_model: Type[Module],
                 model_kwargs: Optional[Dict[str,
                                             Any]] = None, *args, **kwargs):
        """Initializes the feature store.

        Args:
        - enc_model: The model to use for encoding queries.
        - model_kwargs: Additional keyword arguments to pass to the encoding model.
        - args and kwargs: Additional arguments to pass to the parent class.
        """
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.enc_model = enc_model(*args, **kwargs).to(self.device)
        self.enc_model.eval()
        self.model_kwargs = model_kwargs if model_kwargs is not None else dict(
        )
        super().__init__()

    @property
    def x(self) -> Tensor:
        """Returns the node features.
        """
        return self.get_tensor(group_name=None, attr_name='x')

    @property
    def edge_attr(self) -> Tensor:
        """Returns the edge attributes.
        """
        return self.get_tensor(group_name=(None, None), attr_name='edge_attr')

    def retrieve_seed_nodes(self, query: Any, k_nodes: int = 5) -> InputNodes:
        """Retrieves the k_nodes most similar nodes to the given query.

        Args:
        - query: The query to search for.
        - k_nodes: The number of nodes to retrieve (default: 5).

        Returns:
        - The indices of the most similar nodes.
        """
        result = next(self._retrieve_seed_nodes_batch([query], k_nodes))
        gc.collect()
        torch.cuda.empty_cache()
        return result

    def _retrieve_seed_nodes_batch(self, query: Iterable[Any],
                                   k_nodes: int) -> Iterator[InputNodes]:
        """Retrieves the k_nodes most similar nodes to each query in the batch.

        Args:
        - query: The batch of queries to search for.
        - k_nodes: The number of nodes to retrieve.

        Yields:
        - The indices of the most similar nodes for each query.
        """
        if isinstance(self.meta, dict) and self.meta.get("is_hetero", False):
            raise NotImplementedError

        query_enc = self.enc_model.encode(query,
                                          **self.model_kwargs).to(self.device)
        prizes = pairwise_cosine_similarity(query_enc, self.x.to(self.device))
        topk = min(k_nodes, len(self.x))
        for q in prizes:
            _, indices = torch.topk(q, topk, largest=True)
            yield indices

    def retrieve_seed_edges(self, query: Any, k_edges: int = 3) -> InputEdges:
        """Retrieves the k_edges most similar edges to the given query.

        Args:
        - query: The query to search for.
        - k_edges: The number of edges to retrieve (default: 3).

        Returns:
        - The indices of the most similar edges.
        """
        result = next(self._retrieve_seed_edges_batch([query], k_edges))
        gc.collect()
        torch.cuda.empty_cache()
        return result

    def _retrieve_seed_edges_batch(self, query: Iterable[Any],
                                   k_edges: int) -> Iterator[InputEdges]:
        """Retrieves the k_edges most similar edges to each query in the batch.

        Args:
        - query: The batch of queries to search for.
        - k_edges: The number of edges to retrieve.

        Yields:
        - The indices of the most similar edges for each query.
        """
        if isinstance(self.meta, dict) and self.meta.get("is_hetero", False):
            raise NotImplementedError

        query_enc = self.enc_model.encode(query,
                                          **self.model_kwargs).to(self.device)

        prizes = pairwise_cosine_similarity(query_enc,
                                            self.edge_attr.to(self.device))
        topk = min(k_edges, len(self.edge_attr))
        for q in prizes:
            _, indices = torch.topk(q, topk, largest=True)
            yield indices

    def load_subgraph(
        self, sample: Union[SamplerOutput, HeteroSamplerOutput]
    ) -> Union[Data, HeteroData]:
        """Loads a subgraph from the given sample.

        Args:
        - sample: The sample to load the subgraph from.

        Returns:
        - The loaded subgraph.
        """
        if isinstance(sample, HeteroSamplerOutput):
            raise NotImplementedError

        # NOTE: torch_geometric.loader.utils.filter_custom_store can be used here
        # if it supported edge features
        edge_id = sample.edge
        edge_attr = self.edge_attr[edge_id]
        return Data(edge_attr=edge_attr, edge_idx=edge_id)


# TODO: Refactor because composition >> inheritance


def _add_features_to_knn_index(knn_index: ApproxMIPSKNNIndex, emb: Tensor,
                               device: torch.device, batch_size: int = 2**20):
    """Add new features to the existing KNN index in batches.

    Args:
        knn_index (ApproxMIPSKNNIndex): Index to add features to.
        emb (Tensor): Embeddings to add.
        device (torch.device): Device to store in
        batch_size (int, optional): Batch size to iterate by.
            Defaults to 2**20, which equates to 4GB if working with
            1024 dim floats.
    """
    for i in range(0, emb.size(0), batch_size):
        if emb.size(0) - i >= batch_size:
            emb_batch = emb[i:i + batch_size].to(device)
        else:
            emb_batch = emb[i:].to(device)
        knn_index.add(emb_batch)


class ApproxKNNRAGFeatureStore(KNNRAGFeatureStore):
    def __init__(self, enc_model: Type[Module],
                 model_kwargs: Optional[Dict[str,
                                             Any]] = None, *args, **kwargs):
        # TODO: Add kwargs for approx KNN to parameters here.
        super().__init__(enc_model, model_kwargs, *args, **kwargs)
        self.node_knn_index = None
        self.edge_knn_index = None

    def _retrieve_seed_nodes_batch(self, query: Iterable[Any],
                                   k_nodes: int) -> Iterator[InputNodes]:
        if isinstance(self.meta, dict) and self.meta.get("is_hetero", False):
            raise NotImplementedError

        enc_model = self.enc_model.to(self.device)
        query_enc = enc_model.encode(query,
                                     **self.model_kwargs).to(self.device)
        del enc_model
        gc.collect()
        torch.cuda.empty_cache()

        if self.node_knn_index is None:
            self.node_knn_index = ApproxMIPSKNNIndex(num_cells=100,
                                                     num_cells_to_visit=100,
                                                     bits_per_vector=4)
            # Need to add in batches to avoid OOM
            _add_features_to_knn_index(self.node_knn_index, self.x,
                                       self.device)

        output = self.node_knn_index.search(query_enc, k=k_nodes)
        yield from output.index

    def _retrieve_seed_edges_batch(self, query: Iterable[Any],
                                   k_edges: int) -> Iterator[InputEdges]:
        if isinstance(self.meta, dict) and self.meta.get("is_hetero", False):
            raise NotImplementedError

        enc_model = self.enc_model.to(self.device)
        query_enc = enc_model.encode(query,
                                     **self.model_kwargs).to(self.device)
        del enc_model
        gc.collect()
        torch.cuda.empty_cache()

        if self.edge_knn_index is None:
            self.edge_knn_index = ApproxMIPSKNNIndex(num_cells=100,
                                                     num_cells_to_visit=100,
                                                     bits_per_vector=4)
            # Need to add in batches to avoid OOM
            _add_features_to_knn_index(self.edge_knn_index, self.edge_attr,
                                       self.device)

        output = self.edge_knn_index.search(query_enc, k=k_edges)
        yield from output.index


# TODO: These two classes should be refactored
class SentenceTransformerFeatureStore(KNNRAGFeatureStore):
    def __init__(self, *args, **kwargs):
        kwargs['model_name'] = kwargs.get(
            'model_name', 'sentence-transformers/all-roberta-large-v1')
        super().__init__(SentenceTransformer, *args, **kwargs)


class SentenceTransformerApproxFeatureStore(ApproxKNNRAGFeatureStore):
    def __init__(self, *args, **kwargs):
        kwargs['model_name'] = kwargs.get(
            'model_name', 'sentence-transformers/all-roberta-large-v1')
        super().__init__(SentenceTransformer, *args, **kwargs)
