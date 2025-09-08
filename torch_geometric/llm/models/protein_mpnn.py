from itertools import product
from typing import Tuple

import torch
import torch.nn.functional as F

from torch_geometric.nn import knn_graph
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import to_dense_adj, to_dense_batch


class PositionWiseFeedForward(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.out = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_channels),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_channels, in_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(x)


class PositionalEncoding(torch.nn.Module):
    def __init__(self, hidden_channels: int,
                 max_relative_feature: int = 32) -> None:
        super().__init__()
        self.max_relative_feature = max_relative_feature
        self.emb = torch.nn.Embedding(2 * max_relative_feature + 2,
                                      hidden_channels)

    def forward(self, offset, mask) -> torch.Tensor:
        d = torch.clip(offset + self.max_relative_feature, 0,
                       2 * self.max_relative_feature) * mask + (1 - mask) * (
                           2 * self.max_relative_feature + 1)  # noqa: E501
        return self.emb(d.long())


class Encoder(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        dropout: float = 0.1,
        scale: float = 30,
    ) -> None:
        super().__init__()
        self.out_v = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_channels),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_channels, hidden_channels),
        )
        self.out_e = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_channels),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_channels, hidden_channels),
        )
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.dropout3 = torch.nn.Dropout(dropout)
        self.norm1 = torch.nn.LayerNorm(hidden_channels)
        self.norm2 = torch.nn.LayerNorm(hidden_channels)
        self.norm3 = torch.nn.LayerNorm(hidden_channels)
        self.scale = scale
        self.dense = PositionWiseFeedForward(hidden_channels,
                                             hidden_channels * 4)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        # x: [N, d_v]
        # edge_index: [2, E]
        # edge_attr: [E, d_e]
        # update node features
        h_message = self.propagate(x=x, edge_index=edge_index,
                                   edge_attr=edge_attr)
        dh = h_message / self.scale
        x = self.norm1(x + self.dropout1(dh))
        dh = self.dense(x)
        x = self.norm2(x + self.dropout2(dh))
        # update edge features
        row, col = edge_index
        x_i, x_j = x[row], x[col]
        h_e = torch.cat([x_i, x_j, edge_attr], dim=-1)
        h_e = self.out_e(h_e)
        edge_attr = self.norm3(edge_attr + self.dropout3(h_e))
        return x, edge_attr

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        h = torch.cat([x_i, x_j, edge_attr], dim=-1)  # [E, 2*d_v + d_e]
        h = self.out_e(h)  # [E, d_e]
        return h


class Decoder(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        dropout: float = 0.1,
        scale: float = 30,
    ) -> None:
        super().__init__()
        self.out_v = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_channels),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_channels, hidden_channels),
        )
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.norm1 = torch.nn.LayerNorm(hidden_channels)
        self.norm2 = torch.nn.LayerNorm(hidden_channels)
        self.scale = scale
        self.dense = PositionWiseFeedForward(hidden_channels,
                                             hidden_channels * 4)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        x_label: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        # x: [N, d_v]
        # edge_index: [2, E]
        # edge_attr: [E, d_e]
        h_message = self.propagate(x=x, x_label=x_label, edge_index=edge_index,
                                   edge_attr=edge_attr, mask=mask)
        dh = h_message / self.scale
        x = self.norm1(x + self.dropout1(dh))
        dh = self.dense(x)
        x = self.norm2(x + self.dropout2(dh))
        return x

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor,
                x_label_j: torch.Tensor, edge_attr: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        h_1 = torch.cat([x_j, edge_attr, x_label_j], dim=-1)
        h_0 = torch.cat([x_j, edge_attr, torch.zeros_like(x_label_j)], dim=-1)
        h = h_1 * mask + h_0 * (1 - mask)
        h = torch.concat([x_i, h], dim=-1)
        h = self.out_v(h)
        return h


class ProteinMPNN(torch.nn.Module):
    r"""The ProteinMPNN model from the `"Robust deep learning--based
    protein sequence design using ProteinMPNN"
    <https://www.biorxiv.org/content/10.1101/2022.06.03.494563v1>`_ paper.

    Args:
        hidden_dim (int): Hidden channels.
            (default: :obj:`128`)
        num_encoder_layers (int): Number of encode layers.
            (default: :obj:`3`)
        num_decoder_layers (int): Number of decode layers.
            (default: :obj:`3`)
        num_neighbors (int): Number of neighbors for each atom.
            (default: :obj:`30`)
        num_rbf (int): Number of radial basis functions.
            (default: :obj:`16`)
        dropout (float): Dropout rate.
            (default: :obj:`0.1`)
        augment_eps (float): Augmentation epsilon for input coordinates.
            (default: :obj:`0.2`)
        num_positional_embedding (int): Number of positional embeddings.
            (default: :obj:`16`)
        vocab_size (int): Number of vocabulary.
            (default: :obj:`21`)

    .. note::
        For an example of using :class:`ProteinMPNN`, see
        `examples/llm/protein_mpnn.py <https://github.com/pyg-team/
        pytorch_geometric/blob/master/examples/llm/protein_mpnn.py>`_.
    """
    def __init__(
        self,
        hidden_dim: int = 128,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        num_neighbors: int = 30,
        num_rbf: int = 16,
        dropout: float = 0.1,
        augment_eps: float = 0.2,
        num_positional_embedding: int = 16,
        vocab_size: int = 21,
    ) -> None:
        super().__init__()
        self.augment_eps = augment_eps
        self.hidden_dim = hidden_dim
        self.num_neighbors = num_neighbors
        self.num_rbf = num_rbf
        self.embedding = PositionalEncoding(num_positional_embedding)
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(num_positional_embedding + 400, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )
        self.label_embedding = torch.nn.Embedding(vocab_size, hidden_dim)
        self.encoder_layers = torch.nn.ModuleList([
            Encoder(hidden_dim * 3, hidden_dim, dropout)
            for _ in range(num_encoder_layers)
        ])

        self.decoder_layers = torch.nn.ModuleList([
            Decoder(hidden_dim * 4, hidden_dim, dropout)
            for _ in range(num_decoder_layers)
        ])
        self.output = torch.nn.Linear(hidden_dim, vocab_size)

        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def _featurize(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        N, Ca, C, O = (x[:, i, :] for i in range(4))  # noqa: E741
        b = Ca - N
        c = C - Ca
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + Ca

        valid_mask = mask.bool()
        valid_Ca = Ca[valid_mask]
        valid_batch = batch[valid_mask]

        edge_index = knn_graph(valid_Ca, k=self.num_neighbors,
                               batch=valid_batch, loop=True)

        row, col = edge_index
        original_indices = torch.arange(Ca.size(0),
                                        device=x.device)[valid_mask]
        edge_index_original = torch.stack(
            [original_indices[row], original_indices[col]], dim=0)
        row, col = edge_index_original

        rbf_all = []
        for A, B in list(product([N, Ca, C, O, Cb], repeat=2)):
            distances = torch.sqrt(torch.sum((A[row] - B[col])**2, 1) + 1e-6)
            rbf = self._rbf(distances)
            rbf_all.append(rbf)

        return edge_index_original, torch.cat(rbf_all, dim=-1)

    def _rbf(self, D: torch.Tensor) -> torch.Tensor:
        D_min, D_max, D_count = 2., 22., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=D.device)
        D_mu = D_mu.view([1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
        return RBF

    def forward(
        self,
        x: torch.Tensor,
        chain_seq_label: torch.Tensor,
        mask: torch.Tensor,
        chain_mask_all: torch.Tensor,
        residue_idx: torch.Tensor,
        chain_encoding_all: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        device = x.device
        if self.training and self.augment_eps > 0:
            x = x + self.augment_eps * torch.randn_like(x)

        edge_index, edge_attr = self._featurize(x, mask, batch)

        row, col = edge_index
        offset = residue_idx[row] - residue_idx[col]
        # find self vs non-self interaction
        e_chains = ((chain_encoding_all[row] -
                     chain_encoding_all[col]) == 0).long()
        e_pos = self.embedding(offset, e_chains)
        h_e = self.edge_mlp(torch.cat([edge_attr, e_pos], dim=-1))
        h_v = torch.zeros(x.size(0), self.hidden_dim, device=x.device)

        # encoder
        for encoder in self.encoder_layers:
            h_v, h_e = encoder(h_v, edge_index, h_e)

        # mask
        h_label = self.label_embedding(chain_seq_label)
        batch_chain_mask_all, _ = to_dense_batch(chain_mask_all * mask,
                                                 batch)  # [B, N]
        # 0 - visible - encoder, 1 - masked - decoder
        decoding_order = torch.argsort(
            (batch_chain_mask_all + 1e-4) * (torch.abs(
                torch.randn(batch_chain_mask_all.shape, device=device))))
        mask_size = batch_chain_mask_all.size(1)
        permutation_matrix_reverse = F.one_hot(decoding_order,
                                               num_classes=mask_size).float()
        order_mask_backward = torch.einsum(
            'ij, biq, bjp->bqp',
            1 - torch.triu(torch.ones(mask_size, mask_size, device=device)),
            permutation_matrix_reverse,
            permutation_matrix_reverse,
        )
        adj = to_dense_adj(edge_index, batch)
        mask_attend = order_mask_backward[adj.bool()].unsqueeze(-1)

        # decoder
        for decoder in self.decoder_layers:
            h_v = decoder(
                h_v,
                edge_index,
                h_e,
                h_label,
                mask_attend,
            )

        logits = self.output(h_v)
        return F.log_softmax(logits, dim=-1)
