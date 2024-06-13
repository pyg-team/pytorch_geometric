from collections.abc import Iterable, Iterator
from typing import Any, Dict, Optional, Type, Union

import torch
from torch import Tensor
from torch.nn import Module
from torchmetrics.functional import pairwise_cosine_similarity

from torch_geometric.data import Data, HeteroData
from torch_geometric.distributed import LocalFeatureStore
from torch_geometric.nn.nlp import SentenceTransformer
from torch_geometric.sampler import HeteroSamplerOutput, SamplerOutput
from torch_geometric.typing import InputEdges, InputNodes


# NOTE: Only compatible with Homogeneous graphs for now
class KNNRAGFeatureStore(LocalFeatureStore):
    def __init__(self, enc_model: Type[Module],
                 model_kwargs: Optional[Dict[str,
                                             Any]] = None, *args, **kwargs):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.enc_model = enc_model(*args, **kwargs).to(self.device)
        self.enc_model.eval()
        self.model_kwargs = \
            model_kwargs if model_kwargs is not None else dict()
        super().__init__()

    @property
    def x(self) -> Tensor:
        return self.get_tensor(group_name=None, attr_name='x')

    @property
    def edge_attr(self) -> Tensor:
        return self.get_tensor(group_name=(None, None), attr_name='edge_attr')

    def retrieve_seed_nodes(self, query: Any, k_nodes: int = 5) -> InputNodes:
        return next(self._retrieve_seed_nodes_batch([query], k_nodes))

    def _retrieve_seed_nodes_batch(self, query: Iterable[Any],
                                   k_nodes: int) -> Iterator[InputNodes]:
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
        return next(self._retrieve_seed_edges_batch([query], k_edges))

    def _retrieve_seed_edges_batch(self, query: Iterable[Any],
                                   k_edges: int) -> Iterator[InputEdges]:
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

        if isinstance(sample, HeteroSamplerOutput):
            raise NotImplementedError

        # NOTE: torch_geometric.loader.utils.filter_custom_store can be used
        # here if it supported edge features
        node_id = sample.node
        edge_id = sample.edge
        edge_index = torch.stack((sample.row, sample.col), dim=0)
        x = self.x[node_id]
        edge_attr = self.edge_attr[edge_id]

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                    node_idx=node_id, edge_idx=edge_id)


class SentenceTransformerFeatureStore(KNNRAGFeatureStore):
    def __init__(self, *args, **kwargs):
        super().__init__(SentenceTransformer, *args, **kwargs)
