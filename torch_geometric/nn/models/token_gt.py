import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.utils import unbatch


class TokenGT(nn.Module):
    r"""The Tokenized Graph Transformer (TokenGT) model from the
    `"Pure Transformers are Powerful Graph Learners"
    <https://arxiv.org/pdf/2207.02505>` paper.

    TokenGT models graph data by 1. treating all nodes and edges as independent
    tokens, 2. augmenting said tokens with structural information (node and
    type identifiers), and 3. feeding tokens into a standard multi-head
    self-attention Transformer model.

    Args:
        dim_node (int): The node feature dimension.
        dim_edge (int, optional): The edge feature dimension.
        d_p (int): The dimension of node identifiers.
        d (int): The dimension of output embeddings.
        num_heads (int): The number of heads in the multi-head self-attention
            of the Transformer.
        num_encoder_layers (int): The number of sub-encoder layers in the
            Transformer.
        dim_feedforward (int): The dimension of the feedforward neural network
            in the sub-encoder layers of the Transformer.
        include_graph_token (bool): Whether to include a special [graph_token]
            embedding to use for graph-level tasks.
        is_laplacian_node_ids (bool): Whether the provided node identifiers are
            Laplacian eigenvectors. If :obj:`True`, then, during training, 1.
            the sign of eigenvectors are randomly flipped, and 2. dropout gets
            applied to node identifiers.
        dropout (float): Dropout probability used for both the Transformer
            sub-encoder layers and the Laplacian node identifiers.
        device (torch.device): Accelerator device to use.
        norm_first (bool): Whether to perform layer normalisation prior to
            self-attention in the sub-encoder layers.
        activation (str): The activation function used in the sub-encoder
            layers.
        **transformer_kwargs: (optional): Additional arguments passed to
            :class:`torch.nn.TransformerEncoderLayer` object.
    """
    def __init__(
        self,
        dim_node: int,
        dim_edge: Optional[int],
        d_p: int,
        d: int,
        num_heads: int,
        num_encoder_layers: int,
        dim_feedforward: int,
        include_graph_token: bool = False,
        is_laplacian_node_ids: bool = True,
        dropout: float = 0.1,
        device: torch.device = torch.device("cpu"),
        norm_first: bool = True,
        activation: str = "gelu",
        **transformer_kwargs,
    ):
        super().__init__()
        self._d_p = d_p
        self._d = d
        self._num_encoder_layers = num_encoder_layers
        self._is_laplacian_node_ids = is_laplacian_node_ids
        self._device = device

        self._node_features_enc = nn.Linear(dim_node, d, False, device)
        if dim_edge is not None:
            self._edge_features_enc = nn.Linear(dim_edge, d, False, device)
        else:
            self._edge_features_enc = None
        self._node_id_enc = nn.Linear(d_p * 2, d, False, device)
        self._type_id_enc = nn.Embedding(2, d, device=device)
        if include_graph_token is True:
            self._graph_emb = nn.Embedding(1, d, device=device)
        else:
            self._graph_emb = None

        if is_laplacian_node_ids is True:
            self._node_id_dropout = nn.Dropout(dropout)
        else:
            self._node_id_dropout = None

        # standard encoder-only transformer
        enc_layer = nn.TransformerEncoderLayer(
            d,
            num_heads,
            dim_feedforward,
            dropout,
            batch_first=True,
            device=device,
            norm_first=norm_first,
            activation=activation,
            **transformer_kwargs,
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
        node_ids: Tensor,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""Forward pass that returns embeddings for each input node and
        (optionally) a graph-level embedding for each graph in the input.

        Args:
            x (torch.Tensor): The input node features. Needs to have number of
                channels equal to dim_node.
            edge_index (torch.Tensor): The edge indices.
            edge_attr (torch.Tensor, optional): The edge features. If provided,
                needs to have number of channels equal to dim_edge.
            ptr (torch.Tensor): The pointer vector that provides a cumulative
                sum of each graph's node count. The number of entries is one
                more than the number of input graphs. Note: when providing a
                single graph with (say) 5 nodes as input, set equal to
                torch.tensor([0, 5]).
            batch (torch.Tensor): The batch vector that relates each node to a
                specific graph. The number of entries is equal to the number of
                rows in x. Note: when providing a single graph with (say) 5
                nodes as input, set equal to torch.tensor([0, 0, 0, 0, 0]).
            node_ids (torch.Tensor): Orthonormal node identifiers (needs to
                have number of channels equal to d_p).
        """
        batched_emb, src_key_padding_mask, node_mask = (
            self._get_tokenwise_batched_emb(x, edge_index, edge_attr, ptr,
                                            batch, node_ids))
        if self._graph_emb is not None:
            # append special graph token
            b_s = batched_emb.shape[0]
            graph_emb = self._graph_emb.weight.expand(b_s, 1, -1)
            batched_emb = torch.concat((graph_emb, batched_emb), 1)
            b_t = torch.tensor([False], device=self._device).expand(b_s, -1)
            src_key_padding_mask = torch.concat((b_t, src_key_padding_mask), 1)

        batched_emb = self._encoder(batched_emb, None, src_key_padding_mask)
        if self._graph_emb is not None:
            # grab graph token embedding from each batch
            graph_emb = batched_emb[:, 0, :]
            batched_emb = batched_emb[:, 1:, :]
        else:
            graph_emb = None

        # each batch has node + edge + padded entries;
        # select node emb and collapse into 2d tensor that matches x
        node_emb = batched_emb[node_mask]
        return node_emb, graph_emb

    def _get_tokenwise_batched_emb(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor],
        ptr: Tensor,
        batch: Tensor,
        node_ids: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""Adds node and type identifiers, and batches data due to different
        graphs. Returns batched tokenized embeddings, together with masks.
        """
        if self._is_laplacian_node_ids is True and self.training is True:
            # flip eigenvector signs and apply dropout
            unbatched_node_ids = list(unbatch(node_ids, batch))
            for i in range(len(unbatched_node_ids)):
                sign = -1 + 2 * torch.randint(0, 2, (self._d_p, ),
                                              device=self._device)
                unbatched_node_ids[i] = sign * unbatched_node_ids[i]
                unbatched_node_ids[i] = self._node_id_dropout(
                    unbatched_node_ids[i])
            node_ids = torch.concat(unbatched_node_ids, 0)

        node_emb = self._get_node_token_emb(x, node_ids)
        edge_emb = self._get_edge_token_emb(edge_attr, edge_index, node_ids)

        # combine node + edge tokens,
        # and split graphs into padded batches -> [batch_size, max_tokens, d]
        n_nodes = ptr[1:] - ptr[:-1]
        n_edges = self._get_n_edges(edge_index, ptr)
        n_tokens = n_nodes + n_edges
        batched_emb = self._get_batched_emb(
            node_emb,
            edge_emb,
            ptr,
            edge_index,
            n_tokens,
        )

        # construct self-attention and node masks
        src_key_padding_mask = self._get_src_key_padding_mask(n_tokens)
        node_mask = self._get_node_mask(n_tokens, n_nodes)

        return batched_emb, src_key_padding_mask, node_mask

    def _get_node_token_emb(self, x: Tensor, node_ids: Tensor) -> Tensor:
        r"""Applies linear projection to node features, and adds projected node
        identifiers and type identifiers.

        node token embedding = x_prj + node_ids_prj + type_ids
        """
        x_prj = self._node_features_enc(x)
        node_ids_prj = self._node_id_enc(torch.concat((node_ids, node_ids), 1))
        total_nodes = x.shape[0]
        type_ids = self._type_id_enc.weight[0].expand(total_nodes, -1)

        node_emb = x_prj + node_ids_prj + type_ids
        return node_emb  # [total_nodes, d]

    def _get_edge_token_emb(
        self,
        edge_attr: Optional[Tensor],
        edge_index: Tensor,
        node_ids: Tensor,
    ) -> Tensor:
        r"""Applies linear projection to edge features (if present), and adds
        projected node identifiers and type identifiers.

        edge token embedding = edge_attr_prj + node_ids_prj + type_ids
        """
        if edge_attr is not None:
            edge_attr_prj = self._edge_features_enc(edge_attr)
        else:
            edge_attr_prj = None
        node_ids_concat = torch.concat(
            (node_ids[edge_index[0]], node_ids[edge_index[1]]), 1)
        node_ids_prj = self._node_id_enc(node_ids_concat)
        total_edges = edge_index.shape[1]
        type_ids = self._type_id_enc.weight[1].expand(total_edges, -1)

        edge_emb = node_ids_prj + type_ids
        if edge_attr_prj is not None:
            edge_emb = edge_emb + edge_attr_prj
        return edge_emb  # [total_edges, d]

    @staticmethod
    @torch.no_grad()
    def _get_batched_emb(
        node_emb: Tensor,
        edge_emb: Tensor,
        ptr: Tensor,
        edge_index: Tensor,
        n_tokens: Tensor,
    ) -> Tensor:
        r"""Combines node and edge embeddings of each input graph, and pads the
        time dimension to equal that of the input graph with the most nodes +
        edges.
        """
        max_tokens = n_tokens.max().item()
        batch_size = n_tokens.shape[0]
        batched_emb = []
        for i in range(batch_size):
            graph_node_emb = node_emb[ptr[i]:ptr[i + 1]]
            graph_edge_emb = edge_emb[(edge_index[0] >= ptr[i])
                                      & (edge_index[0] < ptr[i + 1])]
            unpadded_emb = torch.concat((graph_node_emb, graph_edge_emb), 0)
            pad = (0, 0, 0, max_tokens - n_tokens[i])
            padded_emb = F.pad(unpadded_emb, pad, value=0.0).unsqueeze(0)
            batched_emb.append(padded_emb)
        return torch.concat(batched_emb, 0)  # [b, t, c]

    @torch.no_grad()
    def _get_src_key_padding_mask(self, n_tokens: Tensor) -> Tensor:
        r"""Computes the src_key_padding_mask tensor which identifies the
        padded values of each batch entry. Passed to the `forward` method of
        :class:`torch.nn.TransformerEncoderLayer`.
        """
        n_batches = len(n_tokens)
        token_index = torch.arange(
            n_tokens.max().item(),
            dtype=torch.long,
            device=self._device,
        ).unsqueeze(0).expand(n_batches, -1)
        src_key_padding_mask = ~torch.less(token_index, n_tokens.unsqueeze(1))
        return src_key_padding_mask

    @torch.no_grad()
    def _get_node_mask(self, n_tokens: Tensor, n_nodes: Tensor) -> Tensor:
        r"""Computes the node mask that gets applied to batches of padded node
        and edge embeddings to a 2d tensor of node embeddings which matches the
        input node features.
        """
        n_batches = len(n_tokens)
        token_index = torch.arange(
            n_tokens.max().item(),
            dtype=torch.long,
            device=self._device,
        ).unsqueeze(0).expand(n_batches, -1)
        node_mask = torch.less(token_index, n_nodes.unsqueeze(1))
        return node_mask

    @torch.no_grad()
    def _get_n_edges(self, edge_index: Tensor, ptr: Tensor) -> Tensor:
        r"""Computes tensor of integers where entries count the number of edges
        belonging to each input graph.
        """
        n_edges = torch.tensor(
            [((edge_index[0] >= ptr[i]) & (edge_index[0] < ptr[i + 1])).sum()
             for i in range(ptr.shape[0] - 1)], device=self._device)
        return n_edges

    def reset_params(self) -> None:
        r"""Resets all learnable parameters of the module."""
        for layer in self._encoder.layers:
            for module in layer.modules():
                if isinstance(module, nn.Linear):
                    module.reset_parameters()
                elif isinstance(module, nn.LayerNorm):
                    module.reset_parameters()
        # reinitialise parameters
        self.apply(lambda m: self._init_params(m, self._num_encoder_layers))

    @staticmethod
    def _init_params(module: nn.Module, layers: int) -> None:
        # modified from https://github.com/jw9730/tokengt/blob/main/
        # large-scale-regression/tokengt/modules/tokenizer.py
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(layers))
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._d})"
