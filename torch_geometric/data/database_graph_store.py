from abc import ABC, abstractmethod
from typing import Any, Optional

from torch_geometric.data.graph_store import GraphStore


class DatabaseGraphStore(GraphStore, ABC):
    r"""Abstract :class:`GraphStore` backed by a database or sampling service.

    Database stores cannot cheaply materialise the full edge index; sampling
    is pushed to the backend instead. Subclasses implement one hook,
    :meth:`query_db`, which executes a backend query and returns the raw
    record. Decoding into PyG tensors is owned by
    code that defines the query and calls :meth:`query_db` (e.g. samplers).

    Args:
        edge_attr_cls (EdgeAttr, optional): User-defined :class:`EdgeAttr`
            subclass. (default: :obj:`None`)
    """
    def __init__(self, edge_attr_cls: Optional[Any] = None):
        super().__init__(edge_attr_cls=edge_attr_cls)

    @abstractmethod
    def query_db(self, query: str, kwargs: dict) -> Any:
        r"""Execute ``query`` against the backend and return the raw result.

        Args:
            query (str): Backend query string (e.g. Cypher).
            kwargs (dict): Query parameters.

        Returns:
            The raw database response.
        """
