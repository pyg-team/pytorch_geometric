import gc
from collections.abc import Iterable, Iterator
from typing import Any, Dict, List, Tuple, Union

import torch
from torch import Tensor

from torch_geometric.data import Data, HeteroData
from torch_geometric.distributed import LocalFeatureStore
from torch_geometric.llm.utils.backend_utils import batch_knn
from torch_geometric.sampler import HeteroSamplerOutput, SamplerOutput
from torch_geometric.typing import InputNodes


# NOTE: Only compatible with Homogeneous graphs for now
class KNNRAGFeatureStore(LocalFeatureStore):
    """A feature store that uses a KNN-based retrieval."""
    def __init__(self) -> None:
        """Initializes the feature store."""
        # to be set by the config
        self.encoder_model = None
        self.k_nodes = None
        self._config: Dict[str, Any] = {}
        super().__init__()

    @property
    def config(self) -> Dict[str, Any]:
        """Get the config for the feature store."""
        return self._config

    def _set_from_config(self, config: Dict[str, Any], attr_name: str) -> None:
        """Set an attribute from the config.

        Args:
            config (Dict[str, Any]): Config dictionary
            attr_name (str): Name of attribute to set

        Raises:
            ValueError: If required attribute not found in config
        """
        if attr_name not in config:
            raise ValueError(
                f"Required config parameter '{attr_name}' not found")
        setattr(self, attr_name, config[attr_name])

    @config.setter  # type: ignore
    def config(self, config: Dict[str, Any]) -> None:
        """Set the config for the feature store.

        Args:
            config (Dict[str, Any]):
                Config dictionary containing required parameters

        Raises:
            ValueError: If required parameters missing from config
        """
        self._set_from_config(config, "k_nodes")
        self._set_from_config(config, "encoder_model")
        assert self.encoder_model is not None, \
            "Need to define encoder model from config"
        self.encoder_model.eval()

        self._config = config

    @property
    def x(self) -> Tensor:
        """Returns the node features."""
        return Tensor(self.get_tensor(group_name=None, attr_name='x'))

    @property
    def edge_attr(self) -> Tensor:
        """Returns the edge attributes."""
        return Tensor(
            self.get_tensor(group_name=(None, None), attr_name='edge_attr'))

    def retrieve_seed_nodes(  # noqa: D417
            self, query: Union[str, List[str],
                               Tuple[str]]) -> Tuple[InputNodes, Tensor]:
        """Retrieves the k_nodes most similar nodes to the given query.

        Args:
            query (Union[str, List[str], Tuple[str]]): The query
                or list of queries to search for.

        Returns:
            The indices of the most similar nodes and the encoded query
        """
        if not isinstance(query, (list, tuple)):
            query = [query]
        assert self.k_nodes is not None, "please set k_nodes via config"
        if len(query) == 1:
            result, query_enc = next(
                self._retrieve_seed_nodes_batch(query, self.k_nodes))
            gc.collect()
            torch.cuda.empty_cache()
            return result, query_enc
        else:
            out_dict = {}
            for i, out in enumerate(
                    self._retrieve_seed_nodes_batch(query, self.k_nodes)):
                out_dict[query[i]] = out
            gc.collect()
            torch.cuda.empty_cache()
            return out_dict

    def _retrieve_seed_nodes_batch(  # noqa: D417
            self, query: Iterable[Any],
            k_nodes: int) -> Iterator[Tuple[InputNodes, Tensor]]:
        """Retrieves the k_nodes most similar nodes to each query in the batch.

        Args:
        - query (Iterable[Any]: The batch of queries to search for.
        - k_nodes (int): The number of nodes to retrieve.

        Yields:
        - The indices of the most similar nodes for each query.
        """
        if isinstance(self.meta, dict) and self.meta.get("is_hetero", False):
            raise NotImplementedError
        assert self.encoder_model is not None, \
            "Need to define encoder model from config"
        query_enc = self.encoder_model.encode(query)
        return batch_knn(query_enc, self.x, k_nodes)

    def load_subgraph(  # noqa
        self,
        sample: Union[SamplerOutput, HeteroSamplerOutput],
        induced: bool = True,
    ) -> Union[Data, HeteroData]:
        """Loads a subgraph from the given sample.

        Args:
            sample: The sample to load the subgraph from.
            induced: Whether to return the induced subgraph.
                Resets node and edge ids.

        Returns:
            The loaded subgraph.
        """
        if isinstance(sample, HeteroSamplerOutput):
            raise NotImplementedError
        """
        NOTE: torch_geometric.loader.utils.filter_custom_store
        can be used here if it supported edge features.
        """
        edge_id = sample.edge
        x = self.x[sample.node]
        edge_attr = self.edge_attr[edge_id]

        edge_idx = torch.stack(
            [sample.row, sample.col], dim=0) if induced else torch.stack(
                [sample.global_row, sample.global_col], dim=0)
        result = Data(x=x, edge_attr=edge_attr, edge_index=edge_idx)

        # useful for tracking what subset of the graph was sampled
        result.node_idx = sample.node
        result.edge_idx = edge_id

        return result


"""
TODO: make class CuVSKNNRAGFeatureStore(KNNRAGFeatureStore)
include a approximate knn flag for the CuVS.
Connect this with a CuGraphGraphStore
for enabling a accelerated boolean flag for RAGQueryLoader.
On by default if CuGraph+CuVS avail.
If not raise note mentioning its speedup.
"""
