import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn.attention.performer import orthogonal_matrix
from torch_geometric.utils import (
    get_laplacian,
    to_dense_adj,
    unbatch_edge_index,
)


class TokenGT(nn.Module):
    def __init__(
            self,
            dim_node_features: int,
            dim_node_identifier: int,
            dim_embedding: int,
            num_heads: int,
            num_encoder_layers: int,
            dim_feedforward: int,
            dim_edge_features: Optional[int] = None,
            include_graph_token: bool = False,
            use_lap_node_identifiers: bool = True,
            dropout: float = 0.1,
            device: torch.device = torch.device("cpu"),
            **kwargs,
    ):
        super().__init__()
        self._dim_node_identifier = dim_node_identifier
        self._dim_embedding = dim_embedding
        self._num_encoder_layers = num_encoder_layers
        self._use_lap_node_identifiers = use_lap_node_identifiers
        self._device = device

        self._node_features_enc = nn.Linear(dim_node_features, dim_embedding,
                                            bias=False, device=device)
        if dim_edge_features is not None:
            self._edge_features_enc = nn.Linear(dim_edge_features,
                                                dim_embedding, bias=False,
                                                device=device)
        else:
            self._edge_features_enc = None
        self._node_id_enc = nn.Linear(dim_node_identifier * 2, dim_embedding,
                                      bias=False, device=device)
        self._type_id_enc = nn.Embedding(2, dim_embedding, device=device)
        if include_graph_token:
            self._graph_emb = nn.Embedding(1, dim_embedding, device=device)
        else:
            self._graph_emb = None

        # standard encoder-only transformer
        enc_layer = nn.TransformerEncoderLayer(
            dim_embedding,
            num_heads,
            dim_feedforward,
            dropout,
            batch_first=True,
            device=device,
            **kwargs,
        )
        self._encoder = nn.TransformerEncoder(enc_layer, num_encoder_layers)

        # initialise parameters
        self.apply(lambda m: self._init_params(m, num_encoder_layers))

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor],
        ptr: Tensor,
        batch: Tensor,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        batched_emb, src_key_padding_mask, node_mask = (
            self._get_tokenwise_batched_emb(x, edge_index, edge_attr, ptr,
                                            batch))
        if self._graph_emb is not None:
            # append special graph token
            b_s = batched_emb.shape[0]
            graph_emb = self._graph_emb.weight.expand(b_s, 1, -1)
            batched_emb = torch.concat((graph_emb, batched_emb), dim=1)
            bool_t = torch.tensor([False], device=self._device).expand(b_s, -1)
            src_key_padding_mask = torch.concat((bool_t, src_key_padding_mask),
                                                dim=1)

        batched_emb = self._encoder(batched_emb, None, src_key_padding_mask)
        if self._graph_emb is not None:
            # grab graph token embedding from each batch
            graph_emb = batched_emb[:, 0, :]
            batched_emb = batched_emb[:, 1:, :]
        else:
            graph_emb = None
        # each batch has node + edge + padded embedding;
        # select node and collapse into 2d tensor
        node_emb = batched_emb[node_mask]
        return node_emb, graph_emb

    def _get_tokenwise_batched_emb(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor],
        ptr: Tensor,
        batch: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        n_nodes = ptr[1:] - ptr[:-1]
        unbatched_edge_indices = unbatch_edge_index(
            edge_index, batch)  # edge_index must be sorted

        # node and edge token embeddings
        if self._use_lap_node_identifiers:
            node_ids = self._get_lap_node_ids(unbatched_edge_indices)
        else:
            node_ids = self._get_orf_node_ids(n_nodes)
        node_emb = self._get_node_token_emb(x, node_ids)
        edge_emb = self._get_edge_token_emb(edge_attr, edge_index, node_ids)

        # combine node + edge tokens; split into padded batches -> (b, t, c)
        n_edges = self._get_n_edges(unbatched_edge_indices)
        n_tokens = n_nodes + n_edges
        edge_ptr = (torch.concat((torch.tensor([0], dtype=torch.long,
                                               device=self._device), n_edges),
                                 dim=0).cumsum(dim=0))
        batched_emb = self._get_batched_emb(
            node_emb,
            edge_emb,
            ptr,
            n_tokens,
            edge_ptr,
        )

        # get attention and node masks
        src_key_padding_mask = self._get_src_key_padding_mask(n_tokens)
        node_mask = self._get_node_mask(n_tokens, n_nodes)

        return batched_emb, src_key_padding_mask, node_mask

    def _get_node_token_emb(self, x: Tensor, node_ids: Tensor) -> Tensor:
        # node token embedding: x_proj + node_id_proj + type_id
        total_nodes = x.shape[0]
        node_emb = (self._node_features_enc(x) +
                    self._node_id_enc(torch.concat(
                        (node_ids, node_ids), dim=1)) + self._type_id_enc(
                            torch.zeros(total_nodes, dtype=torch.long,
                                        device=self._device))
                    )  # (total_nodes, dim_embedding)
        return node_emb

    def _get_edge_token_emb(
        self,
        edge_attr: Optional[Tensor],
        edge_index: Tensor,
        node_ids: Tensor,
    ) -> Tensor:
        # edge token embedding: edge_attr_proj + node_id_proj + type_id
        total_edges = edge_index.shape[1]
        edge_attr_proj = (self._edge_features_enc(edge_attr)
                          if edge_attr is not None else torch.zeros(
                              (total_edges,
                               self._dim_embedding), device=self._device))
        node_ids_proj = self._node_id_enc(
            torch.concat((node_ids[edge_index[0]], node_ids[edge_index[1]]),
                         dim=1))
        edge_emb = (edge_attr_proj + node_ids_proj + self._type_id_enc(
            torch.ones(total_edges, dtype=torch.long, device=self._device))
                    )  # (total_edges, dim_embedding)
        return edge_emb

    @staticmethod
    def _get_batched_emb(
        node_emb: Tensor,
        edge_emb: Tensor,
        ptr: Tensor,
        n_tokens: Tensor,
        edge_ptr: Tensor,
    ):
        max_tokens = n_tokens.max().item()
        batch_size = len(n_tokens)

        batched_tokens = torch.concat([
            F.pad(
                torch.concat((node_emb[ptr[i]:ptr[i + 1]],
                              edge_emb[edge_ptr[i]:edge_ptr[i + 1]]), dim=0),
                (0, 0, 0, max_tokens - n_tokens[i]), value=0.0).unsqueeze(0)
            for i in range(batch_size)
        ], dim=0)
        return batched_tokens

    def _get_src_key_padding_mask(self, n_tokens: Tensor) -> Tensor:
        n_batches = len(n_tokens)
        token_index = torch.arange(
            n_tokens.max().item(),
            dtype=torch.long,
            device=self._device,
        ).unsqueeze(0).expand(n_batches, -1)
        src_key_padding_mask = ~torch.less(token_index, n_tokens.unsqueeze(1))
        return src_key_padding_mask

    def _get_node_mask(self, n_tokens: Tensor, n_nodes: Tensor) -> Tensor:
        n_batches = len(n_tokens)
        token_index = torch.arange(
            n_tokens.max().item(),
            dtype=torch.long,
            device=self._device,
        ).unsqueeze(0).expand(n_batches, -1)
        node_mask = torch.less(token_index, n_nodes.unsqueeze(1))
        return node_mask

    def _get_n_edges(self, unbatched_edge_indices: List[Tensor]) -> Tensor:
        n_edges = torch.tensor([
            unbatched_edge_index.shape[1]
            for unbatched_edge_index in unbatched_edge_indices
        ], dtype=torch.long, device=self._device)
        return n_edges

    def _get_orf_node_ids(self, n_nodes: Tensor) -> Tensor:
        """Generate ORF independently for each graph; concat into 2d tensor."""
        orf_node_ids = []
        for n in n_nodes:
            orth_mat = orthogonal_matrix(n, n)
            orth_mat = self._get_reshaped_orth_mat(orth_mat, n)
            orf_node_ids.append(orth_mat)
        orf = torch.concat(orf_node_ids, dim=0).to(self._device)
        orf = F.normalize(orf, p=2, dim=1)
        return orf

    def _get_lap_node_ids(self,
                          unbatched_edge_indices: List[Tensor]) -> Tensor:
        """Generate laplacian eigenvectors independently for each graph,
        then concat into 2d tensor.
        """
        lap_node_ids = []
        for unbatched_edge_index in unbatched_edge_indices:
            lap_edge_index, lap_edge_attr = get_laplacian(
                unbatched_edge_index, normalization="sym")
            lap_mat = to_dense_adj(edge_index=lap_edge_index,
                                   edge_attr=lap_edge_attr)[0]
            _, eigenvectors = torch.linalg.eigh(lap_mat)

            n = eigenvectors.shape[0]
            orth_mat = self._get_reshaped_orth_mat(eigenvectors, n)
            lap_node_ids.append(orth_mat)
        return torch.concat(lap_node_ids, dim=0)

    def _get_reshaped_orth_mat(self, orth_mat: Tensor, n: int) -> Tensor:
        if n < self._dim_node_identifier:
            orth_mat = F.pad(orth_mat, (0, self._dim_node_identifier - n),
                             value=0.0)
        else:
            orth_mat = orth_mat[:, :self._dim_node_identifier]
        return orth_mat

    def reset_params(self) -> None:
        super().reset_parameters()
        self._encoder.reset_parameters()

        # reinitialise parameters
        self.apply(lambda m: self._init_params(m, self._num_encoder_layers))

    @staticmethod
    def _init_params(module, num_encoder_layers) -> None:
        # modified from https://github.com/jw9730/tokengt/blob/main/
        # large-scale-regression/tokengt/modules/tokenizer.py
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=0.0, std=0.02 / math.sqrt(num_encoder_layers))
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
