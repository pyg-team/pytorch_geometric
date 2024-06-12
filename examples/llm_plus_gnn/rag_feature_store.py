from typing import Iterable, Type, Optional, Dict, Any

import torch
from torch import Tensor
from torch.nn import Module
from torchmetrics.functional import pairwise_cosine_similarity

from torch_geometric.distributed import LocalFeatureStore
from torch_geometric.nn.nlp import SentenceTransformer


# NOTE: Only compatible with Homogeneous graphs for now
class KNNRAGFeatureStore(LocalFeatureStore):
    def __init__(self, enc_model: Type[Module], model_kwargs: Optional[Dict[str, Any]] = None, *args, **kwargs):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.enc_model = enc_model(*args, **kwargs).to(self.device)
        self.enc_model.eval()
        self.model_kwargs = model_kwargs if model_kwargs is not None else dict()
        super().__init__()

    @property
    def x(self) -> Tensor:
        return self.get_tensor(group_name=None, attr_name='x')

    @property
    def edge_attr(self) -> Tensor:
        return self.get_tensor(group_name=(None, None), attr_name='edge_attr')

    def retrieve_seed_nodes(self, query: Iterable[str],
                            k_nodes: int = 5) -> Iterable[Tensor]:
        query_enc = self.enc_model.encode(query, **self.model_kwargs).to(self.device)
        prizes = pairwise_cosine_similarity(query_enc, self.x.to(self.device))
        topk = min(k_nodes, len(self.x))
        topk_n_indices = []
        for q in prizes:
            _, indices = torch.topk(q, topk, largest=True)
            topk_n_indices.append(indices)

        return topk_n_indices

    def retrieve_seed_edges(self, query: Iterable[str],
                            k_edges: int = 3) -> Iterable[Tensor]:
        query_enc = self.enc_model.encode(query, **self.model_kwargs).to(self.device)
        prizes = pairwise_cosine_similarity(query_enc,
                                            self.edge_attr.to(self.device))
        topk = min(k_edges, len(self.edge_attr))
        topk_e_indices = []
        for q in prizes:
            _, indices = torch.topk(q, topk, largest=True)
            topk_e_indices.append(indices) 

        return topk_e_indices


class SentenceTransformerFeatureStore(KNNRAGFeatureStore):
    def __init__(self, *args, **kwargs):
        super().__init__(SentenceTransformer, *args, **kwargs)
