"""
This implementation is based on both the original paper's codebase <https://github.com/jw9730/tokengt> and Michail Melonas' implementation <https://github.com/pyg-team/pytorch_geometric/pull/9834>.
"""

from typing import Literal, Optional, Tuple

from torch import Tensor
import torch
import torch.nn as nn
from torch.nn.modules.fold import F

from torch_geometric.nn.attention.performer import orthogonal_matrix


class TokenGT(nn.Module):
    r"""The Tokenized Graph Transformer (TokenGT) model from the
    `"Pure Transformers are Powerful Graph Learners"
    <https://arxiv.org/pdf/2207.02505>` paper.

    TokenGT models graph data by 1. treating all nodes and edges as independent
    tokens, 2. augmenting said tokens with structural information (node and
    type identifiers), and 3. feeding tokens into a standard multi-head
    self-attention Transformer model. The Transformer retrieves structural information
    based on orthogonal vectors called node identifiers. There are two ways to generate node identifiers:
    - "orf": Use random orthogonal features as node identifiers. These are generated anew for every pass to ensure randomization.
    - "laplacian": Use Laplacian eigenvectors as node identifiers (to be used in combination with the AddLaplacianNodeIdentifiers transform).

    Args:
        num_atoms (int): The number of unique atom (node) types. Note that TokenGT expects the node features to have at most one dimension. Multiple dimensions should be collapsed into a single dimension by summing them up with an appropriate offset.
        num_edges (int): The number of unique edge types. Note that TokenGT expects the edge features to have at most one dimension. Multiple dimensions should be collapsed into a single dimension by summing them up with an appropriate offset.
        node_id_mode (Literal["orf", "laplacian"]): Determines how node structural identifiers are generated.
            "orf": Use random orthogonal features as node identifiers.
            "laplacian": Use Laplacian eigenvectors as node identifiers (to be used in combination with the AddLaplacianNodeIdentifiers transform).
        d_p (int): The dimension of node identifiers.
        num_encoder_layers (int): Number of Transformer encoder layers.
        embedding_dim (int): Dimension of the token embeddings
        ffn_embedding_dim (int): Dimension of the feedforward layers in the Transformer.
        num_attention_heads (int): Number of attention heads in the Transformer.
        dropout (float, optional): Dropout probability passed to the Transformer. (Default: 0.1)
        lap_node_id_eig_dropout (float, optional): Dropout probability for Laplacian eigenvectors. (Default: 0.0)
        lap_node_id_sign_flip (bool, optional): Whether to apply random sign flipping to Laplacian eigenvectors during training. (Default: False)
        encoder_normalize_before (bool, optional): If True, apply LayerNorm before passing tokens to the Transformer encoder. (Default: False)
        norm_first (bool, optional): If True, apply LayerNorm before attention and feedforward modules. (Default: False)
        activation_fn (str, optional): Activation function to use in the feedforward network. (Default: "gelu")
        **transformer_kwargs: Additional arguments for the TransformerEncoderLayer.
    """

    def __init__(
        self,
        num_atoms: int,
        num_edges: int,
        node_id_mode: Literal["orf", "laplacian"],
        d_p: int,
        num_encoder_layers: int,
        embedding_dim: int,
        ffn_embedding_dim: int,
        num_attention_heads: int,
        dropout: float = 0.0,
        lap_node_id_eig_dropout: float = 0.0,
        lap_node_id_sign_flip: bool = False,
        encoder_normalize_before: bool = False,
        norm_first: bool = False,
        activation_fn: str = "gelu",
        **transformer_kwargs,
    ) -> None:

        super().__init__()

        assert node_id_mode in [
            "orf",
            "laplacian",
        ], "node_id_mode must be either 'orf' or 'laplacian'"

        self.embedding_dim = embedding_dim
        self.node_id_mode = node_id_mode
        self.d_p = d_p
        self.lap_node_id_sign_flip = lap_node_id_sign_flip

        self.node_feature_encoder = nn.Embedding(
            num_atoms, embedding_dim, padding_idx=0
        )
        self.edge_feature_encoder = nn.Embedding(
            num_edges, embedding_dim, padding_idx=0
        )
        self.graph_token = nn.Embedding(1, embedding_dim)

        self.node_id_encoder = nn.Linear(2 * d_p, embedding_dim, bias=False)
        self.type_id_encoder = nn.Embedding(2, embedding_dim)

        if node_id_mode == "laplacian" and lap_node_id_eig_dropout > 0:
            self.lap_eig_dropout = nn.Dropout2d(p=lap_node_id_eig_dropout)
        else:
            self.lap_eig_dropout = None

        if encoder_normalize_before:
            self.emb_layer_norm = nn.LayerNorm(embedding_dim)
        else:
            self.emb_layer_norm = None

        enc_layer = nn.TransformerEncoderLayer(
            embedding_dim,
            num_attention_heads,
            ffn_embedding_dim,
            dropout,
            batch_first=True,
            norm_first=norm_first,
            activation=activation_fn,
            **transformer_kwargs,
        )
        self._transformer_encoder = nn.TransformerEncoder(enc_layer, num_encoder_layers)

        self.apply(self._init_params)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor],
        ptr: Tensor,
        batch: Tensor,
        node_ids: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""
        Computes the token embeddings for each input node and edge, and passes them through the Transformer encoder. Note that if node_id_mode is "laplacian", node_ids must be provided. They can be generated using the AddLaplacianNodeIdentifiers transform.
        Returns a tuple of the node and edge token embeddings, and the graph-level embedding.
        """

        n_nodes = ptr[1:] - ptr[:-1]

        if self.node_id_mode == "laplacian":
            assert (
                node_ids is not None
            ), "node_ids must be provided when node_id_mode is laplacian"
            assert (
                node_ids.shape[1] == self.d_p
            ), f"node_ids must have {self.d_p} channels"

            if self.lap_eig_dropout is not None:
                node_ids = self.lap_eig_dropout(node_ids[..., None, None]).view(
                    node_ids.size()
                )

            if self.lap_node_id_sign_flip and self.training:
                # Flip the sign of random eigenvectors, i.e. flip the sign of the
                # 2nd dimension of node_ids in the same graph.
                signs = -1 + 2 * torch.randint(
                    0, 2, (len(ptr) - 1, node_ids.shape[1]), device=node_ids.device
                )  # [n_graphs, d_p]
                signs = signs.repeat_interleave(
                    ptr[1:] - ptr[:-1], dim=0
                )  # [n_nodes, d_p]
                node_ids = signs * node_ids

        elif self.node_id_mode == "orf":
            orfs = []
            for i in range(len(n_nodes)):
                orf = orthogonal_matrix(n_nodes[i], n_nodes[i])
                if n_nodes[i] < self.d_p:
                    orf = F.pad(orf, (0, self.d_p - n_nodes[i]), value=0.0)
                else:
                    orf = orf[:, : self.d_p]
                orf = F.normalize(orf, p=2, dim=1)
                orfs.append(orf)
            node_ids = torch.cat(orfs, dim=0)

        token_embs, src_key_padding_mask, node_mask, edge_mask = (
            self.get_token_embeddings(x, edge_index, edge_attr, ptr, batch, node_ids)
        )

        if self.emb_layer_norm is not None:
            token_embs = self.emb_layer_norm(token_embs)

        token_embs = self._transformer_encoder(
            token_embs, src_key_padding_mask=src_key_padding_mask
        )

        graph_rep = token_embs[:, 0, :]
        return token_embs[node_mask], token_embs[edge_mask], graph_rep

    def get_token_embeddings(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor],
        ptr: Tensor,
        batch: Tensor,
        node_ids: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""
        Computes the node and edge token embeddings, and returns a tuple of the padded embeddings, the padding mask, the node mask, and the edge mask.
        """

        n_nodes = ptr[1:] - ptr[:-1]
        batch_size = n_nodes.shape[0]
        n_edges = torch.bincount(batch[edge_index[0]], minlength=batch_size)
        n_tokens = n_nodes + n_edges
        max_tokens = n_tokens.max()

        node_embbeddings = self.get_node_token_embeddings(x, node_ids)
        edge_embeddings = self.get_edge_token_embeddings(
            edge_attr, edge_index, node_ids
        )

        # Pad the embeddings to match the sample with the most tokens in the batch.
        padded_embbeddings = torch.zeros(
            batch_size,
            max_tokens,
            self.embedding_dim,
            device=x.device,
            dtype=node_embbeddings.dtype,
        )

        # Compute node and edge token positions in the padded embeddings.
        token_pos = (
            torch.arange(max_tokens, device=x.device)
            .unsqueeze(0)
            .expand(batch_size, max_tokens)
        )
        padded_node_mask = token_pos < n_nodes.unsqueeze(1)
        padded_edge_mask = torch.logical_and(
            token_pos >= n_nodes.unsqueeze(1), token_pos < n_tokens.unsqueeze(1)
        )

        # Place the node and edge embeddings in the padded embeddings.
        padded_embbeddings[padded_node_mask] = node_embbeddings
        padded_embbeddings[padded_edge_mask] = edge_embeddings

        src_key_padding_mask = ~torch.less(token_pos, n_tokens.unsqueeze(1))

        return (
            padded_embbeddings,
            src_key_padding_mask,
            padded_node_mask,
            padded_edge_mask,
        )

    def get_node_token_embeddings(self, x: Tensor, node_ids: Tensor) -> Tensor:
        r"""Applies linear projection to node features, and adds projected node identifiers and type identifiers.

        node token embedding = x_prj + node_ids_prj + type_ids
        """
        x_prj = self.node_feature_encoder(x).sum(-2)
        node_ids_prj = self.node_id_encoder(torch.concat((node_ids, node_ids), 1))
        total_nodes = x.shape[0]
        type_ids = self.type_id_encoder.weight[0].expand(total_nodes, -1)

        node_emb = x_prj + node_ids_prj + type_ids
        return node_emb  # [total_nodes, embedding_dim]

    def get_edge_token_embeddings(
        self,
        edge_attr: Optional[Tensor],
        edge_index: Tensor,
        node_ids: Tensor,
    ) -> Tensor:
        r"""Applies linear projection to edge features (if present), and adds
        projected node identifiers and type identifiers.

        edge token embeddings = edge_attr_prj + node_ids_prj + type_ids
        """
        node_ids_concat = torch.concat(
            (node_ids[edge_index[0]], node_ids[edge_index[1]]), 1
        )
        node_ids_prj = self.node_id_encoder(node_ids_concat)
        total_edges = edge_index.shape[1]
        type_ids = self.type_id_encoder.weight[1].expand(total_edges, -1)

        edge_emb = node_ids_prj + type_ids
        if edge_attr is not None:
            edge_emb = edge_emb + self.edge_feature_encoder(edge_attr).sum(-2)
        return edge_emb  # [total_edges, embedding_dim]

    @staticmethod
    def _init_params(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
