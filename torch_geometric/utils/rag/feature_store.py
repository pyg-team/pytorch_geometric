import gc
from collections.abc import Iterable, Iterator
from typing import Any, Dict, List, Type, Union, Tuple

import torch
from torch import Tensor
from torch.nn import Module

from torch_geometric.data import Data, HeteroData
from torch_geometric.distributed import LocalFeatureStore
from torch_geometric.nn.pool import ApproxMIPSKNNIndex
from torch_geometric.sampler import HeteroSamplerOutput, SamplerOutput
from torch_geometric.typing import FeatureTensorType, InputNodes
from torch_geometric.utils.rag.backend_utils import batch_knn


# NOTE: Only compatible with Homogeneous graphs for now
class KNNRAGFeatureStore(LocalFeatureStore):
    """A feature store that uses a KNN-based retrieval."""
    def __init__(self) -> None:
        """Initializes the feature store."""
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # to be set by the config
        self.encoder_model = None
        self.k_nodes = None

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

    @config.setter
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
        if self.encoder_model is not None:
            self.encoder_model = self.encoder_model.to(self.device)
            self.encoder_model.eval()

        self._config = config

    @property
    def x(self) -> FeatureTensorType:
        """Returns the node features."""
        return self.get_tensor(group_name=None, attr_name='x')

    @property
    def edge_attr(self) -> FeatureTensorType:
        """Returns the edge attributes."""
        return self.get_tensor(group_name=(None, None), attr_name='edge_attr')

    def retrieve_seed_nodes(
        self, query: Union[str, List[str], Tuple[str]]
    ) -> Tuple[InputNodes, Tensor]:  # noqa
        """Retrieves the k_nodes most similar nodes to the given query.

        Args:
        - query (Union[str, List[str], Tuple[str]]):
            The query or list of queries to search for.

        Returns:
        - The indices of the most similar nodes and the encoded query
        """
        if not isinstance(query, (list, tuple)):
            query = [query]
        result, query_enc = next(
            self._retrieve_seed_nodes_batch(query, self.k_nodes))
        gc.collect()
        torch.cuda.empty_cache()
        return result, query_enc

    def _retrieve_seed_nodes_batch(  # noqa
            self, query: Iterable[Any],
            k_nodes: int) -> Iterator[Tuple[InputNodes, Tensor]]:
        """Retrieves the k_nodes most similar nodes to each query in the batch.

        Args:
        - query: The batch of queries to search for.
        - k_nodes: The number of nodes to retrieve.

        Yields:
        - The indices of the most similar nodes for each query.
        """
        if isinstance(self.meta, dict) and self.meta.get("is_hetero", False):
            raise NotImplementedError

        query_enc = self.encoder_model.encode(query).to(self.device)
        return batch_knn(query_enc, self.x, k_nodes)

    def load_subgraph(  # noqa
        self,
        sample: Union[SamplerOutput, HeteroSamplerOutput],
        induced: bool = True,
    ) -> Union[Data, HeteroData]:
        """Loads a subgraph from the given sample.

        Args:
        - sample: The sample to load the subgraph from.
        - induced: Whether to return the induced subgraph.
            Resets node and edge ids.

        Returns:
        - The loaded subgraph.
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


# TODO: Refactor because composition >> inheritance


def _add_features_to_knn_index(knn_index: ApproxMIPSKNNIndex, emb: Tensor,
                               device: torch.device,
                               batch_size: int = 2**20) -> None:
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
    def __init__(self, encoder_model: Type[Module], *args, **kwargs):
        # TODO: Add kwargs for approx KNN to parameters here.
        super().__init__(encoder_model, *args, **kwargs)
        self.node_knn_index = None
        self.edge_knn_index = None

    def _retrieve_seed_nodes_batch(self, query: Iterable[Any],
                                   k_nodes: int) -> Iterator[InputNodes]:
        if isinstance(self.meta, dict) and self.meta.get("is_hetero", False):
            raise NotImplementedError

        encoder_model = self.encoder_model.to(self.device)
        query_enc = encoder_model.encode(query).to(self.device)
        del encoder_model
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
