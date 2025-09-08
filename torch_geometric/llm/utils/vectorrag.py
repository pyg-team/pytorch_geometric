# mypy: ignore-errors
import os
from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional, Protocol, Union

import torch
from torch import Tensor

from torch_geometric.data import Data
from torch_geometric.llm.models import SentenceTransformer
from torch_geometric.llm.utils.backend_utils import batch_knn


class VectorRetriever(Protocol):
    """Protocol for VectorRAG."""
    @abstractmethod
    def query(self, query: Any, **kwargs: Optional[Dict[str, Any]]) -> Data:
        """Retrieve a context for a given query."""
        ...


class DocumentRetriever(VectorRetriever):
    """Retrieve documents from a vector database."""
    def __init__(self, raw_docs: List[str],
                 embedded_docs: Optional[Tensor] = None, k_for_docs: int = 2,
                 model: Optional[Union[SentenceTransformer, torch.nn.Module,
                                       Callable]] = None,
                 model_kwargs: Optional[Dict[str, Any]] = None):
        """Retrieve documents from a vector database.

        Args:
            raw_docs: List[str]: List of raw documents.
            embedded_docs: Optional[Tensor]: Embedded documents.
            k_for_docs: int: Number of documents to retrieve.
            model: Optional[Union[SentenceTransformer, torch.nn.Module]]:
                Model to use for encoding.
            model_kwargs: Optional[Dict[str, Any]]:
                Keyword arguments to pass to the model.
        """
        self.raw_docs = raw_docs
        self.embedded_docs = embedded_docs
        self.k_for_docs = k_for_docs
        self.model = model

        if self.model is not None:
            self.encoder = self.model
            self.model_kwargs = model_kwargs

        if self.embedded_docs is None:
            assert self.model is not None, \
                "Model must be provided if embedded_docs is not provided"
            self.model_kwargs = model_kwargs or {}
            self.embedded_docs = self.encoder(self.raw_docs,
                                              **self.model_kwargs)
            # we don't want to print the verbose output in `query`
            self.model_kwargs.pop("verbose", None)

    def query(self, query: Union[str, Tensor]) -> List[str]:
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

    def save(self, path: str) -> None:
        """Save the DocumentRetriever instance to disk.

        Args:
            path: str: Path where to save the retriever.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        # Prepare data to save
        save_dict = {
            'raw_docs': self.raw_docs,
            'embedded_docs': self.embedded_docs,
            'k_for_docs': self.k_for_docs,
        }

        # We do not serialize the model
        torch.save(save_dict, path)

    @classmethod
    def load(cls, path: str, model: Union[SentenceTransformer, torch.nn.Module,
                                          Callable],
             model_kwargs: Optional[Dict[str, Any]] = None) -> VectorRetriever:
        """Load a DocumentRetriever instance from disk.

        Args:
            path: str: Path to the saved retriever.
            model: Union[SentenceTransformer, torch.nn.Module, Callable]:
                Model to use for encoding.
                If None, the saved model will be used if available.
            model_kwargs: Optional[Dict[str, Any]]
                Key word args to be passed to model

        Returns:
            DocumentRetriever: The loaded retriever.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No saved document retriever found at {path}")

        save_dict = torch.load(path, weights_only=False)
        if save_dict['embedded_docs'] is not None \
                and isinstance(save_dict['embedded_docs'], Tensor)\
                and model_kwargs is not None:
            model_kwargs.pop("verbose", None)
        # Create a new DocumentRetriever with the loaded data
        return cls(raw_docs=save_dict['raw_docs'],
                   embedded_docs=save_dict['embedded_docs'],
                   k_for_docs=save_dict['k_for_docs'], model=model,
                   model_kwargs=model_kwargs)
