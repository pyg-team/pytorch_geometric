from abc import abstractmethod
from typing import Any, Dict, List, Optional, Protocol, Union

import torch
from torch import Tensor

from torch_geometric.data import Data
from torch_geometric.nn.nlp import SentenceTransformer
from torch_geometric.utils.rag.backend_utils import batch_knn


class VectorRAG(Protocol):
    """Protocol for VectorRAG."""
    @abstractmethod
    def query(self, query: Any, **kwargs) -> Data:
        """Retrieve a context for a given query."""
        ...


class DocumentRetriever(VectorRAG):
    """Retrieve documents from a vector database."""
    def __init__(self, raw_docs: List[str],
                 embedded_docs: Optional[Tensor] = None, k_for_docs: int = 2,
                 model: Optional[Union[SentenceTransformer,
                                       torch.nn.Module]] = None,
                 model_method_to_call: Optional[str] = "encode",
                 model_kwargs: Optional[Dict[str, Any]] = None):
        """Retrieve documents from a vector database.

        Args:
            raw_docs: List[str]: List of raw documents.
            embedded_docs: Optional[Tensor]: Embedded documents.
            k_for_docs: int: Number of documents to retrieve.
            model: Optional[Union[SentenceTransformer, torch.nn.Module]]: Model to use for encoding.
            model_method_to_call: Optional[str]: Method to call on the model.
            model_kwargs: Optional[Dict[str, Any]]: Keyword arguments to pass to the model.
        """
        self.raw_docs = raw_docs
        self.embedded_docs = embedded_docs
        self.k_for_docs = k_for_docs
        self.model = model

        if self.model is not None:
            self.encoder = getattr(self.model, model_method_to_call)
            self.model_kwargs = model_kwargs

        if self.embedded_docs is None:
            assert self.model is not None, "Model must be provided if embedded_docs is not provided"
            self.embedded_docs = self.encoder(self.raw_docs,
                                              **self.model_kwargs)

    def query(self, query: Union[str, Tensor]) -> Data:
        """Retrieve documents from the vector database.

        Args:
            query: Union[str, Tensor]: Query to retrieve documents for.

        Returns:
            List[str]: Documents retrieved from the vector database.
        """
        if isinstance(query, str):
            query_enc = self.encoder(query, **self.model_kwargs)
        else:
            query_enc = query

        selected_doc_idxs, _ = next(
            batch_knn(query_enc, self.embedded_docs, self.k_for_docs))
        return [self.raw_docs[i] for i in selected_doc_idxs]
