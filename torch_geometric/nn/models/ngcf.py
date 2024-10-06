from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn.models.lightgcn import BPRLoss
from torch_geometric.typing import Adj, OptTensor, SparseTensor
from torch_geometric.utils import is_sparse, to_edge_index


class NGCF(nn.Module):
    r"""The NGCF model from the `"Neural Graph Collaborative Filtering"
    <https://arxiv.org/abs/1905.08108>`_ paper.

    :class:`~torch_geometric.nn.models.NGCF` learns embeddings by explicitly
    injecting collaborative signal into the embedding process.

    .. math::
        \mathbf{E}^{(l)} = \text{LeakyReLU} \left( (\mathcal{L} + \mathbf{I})
        \mathbf{E}^{(l-1)} \mathbf{W}_1^{(l)} + \mathcal{L} \mathbf{E}^{(l-1)}
        \odot \mathbf{E}^{(l-1)} \mathbf{W}_2^{(l)} \right)

    Args:
        num_nodes (int): The number of nodes in the graph.
        emb_size (int): The dimensionality of node embeddings.
        num_layers (int): The number of layers for propagation.
        node_dropout (float): The probability to randomly dropout
            edges for message passing. (default: :obj:`0.1`)
    """
    def __init__(
        self,
        num_nodes: int,
        emb_size: int,
        num_layers: int,
        node_dropout: float = 0.1,
    ):
        super().__init__()
        assert node_dropout >= 0 and node_dropout <= 1.0
        self.num_nodes = num_nodes
        self.emb_size = emb_size
        self.num_layers = num_layers
        self.node_dropout = node_dropout
        self.emb = nn.Embedding(num_nodes, emb_size)
        self.gc_layers = nn.ModuleList()
        self.bi_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gc_layers.append(nn.Linear(emb_size, emb_size))
            self.bi_layers.append(nn.Linear(emb_size, emb_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.emb.weight)
        for lin in self.gc_layers:
            nn.init.xavier_uniform_(lin.weight)
        for lin in self.bi_layers:
            nn.init.xavier_uniform_(lin.weight)

    def _sparse_dropout(
        self,
        row: Tensor,
        col: Tensor,
        value: Tensor,
        rate: float,
        nnz: int,
    ) -> SparseTensor:
        rand: Tensor = (1 - rate) + torch.rand(nnz)
        assert isinstance(rand, Tensor)
        dropout_mask = torch.floor(rand).type(torch.bool)
        adj = SparseTensor(
            row=row[dropout_mask],
            col=col[dropout_mask],
            value=value[dropout_mask] * (1. / (1 - rate)),
            sparse_sizes=(self.num_nodes, self.num_nodes),
        )
        return adj

    def get_embedding(
        self,
        edge_index: Adj,
        edge_weight: OptTensor = None,
    ) -> Tensor:
        r"""Returns the embedding of nodes in the graph."""
        ego_emb = self.emb.weight
        all_embs: List[Tensor] = [ego_emb]
        if isinstance(edge_index, Tensor):
            edge_index = SparseTensor(
                row=edge_index[0],
                col=edge_index[1],
                value=edge_weight,
                sparse_sizes=(self.num_nodes, self.num_nodes),
            ).to(edge_index.device)
        if self.node_dropout > 0 and self.training:
            row, col, value = edge_index.coo()
            adj = self._sparse_dropout(
                row,
                col,
                value,
                self.node_dropout,
                edge_index.nnz(),
            )
        else:
            adj = edge_index
        adj = adj.to_device(edge_index.device())
        for i in range(len(self.gc_layers)):
            msg_emb = adj @ ego_emb
            aggr_emb = self.gc_layers[i](msg_emb)
            bi_emb = torch.mul(ego_emb, msg_emb)
            bi_emb = self.bi_layers[i](bi_emb)
            ego_emb = F.leaky_relu(aggr_emb + bi_emb, negative_slope=0.2)
            ego_emb = F.dropout(ego_emb)
            norm_emb = F.normalize(ego_emb, p=2, dim=1)
            all_embs += [norm_emb]
        res_embs: Tensor = torch.cat(all_embs, 1)
        return res_embs

    def recommendation_loss(
        self,
        pos_edge_rank: Tensor,
        neg_edge_rank: Tensor,
        node_id: OptTensor = None,
        lambda_reg: float = 1e-4,
        **kwargs,
    ) -> Tensor:
        r"""Computes the model loss for a ranking objective via the Bayesian
        Personalized Ranking (BPR) loss.

        .. note::

            The i-th entry in the :obj:`pos_edge_rank` vector and i-th entry
            in the :obj:`neg_edge_rank` entry must correspond to ranks of
            positive and negative edges of the same entity (*e.g.*, user).

        Args:
            pos_edge_rank (torch.Tensor): Positive edge rankings.
            neg_edge_rank (torch.Tensor): Negative edge rankings.
            node_id (torch.Tensor, optional): The indices of the nodes involved
                for deriving a prediction for both positive and negative edges.
                If set to :obj:`None`, all nodes will be used.
            lambda_reg (int, optional): The :math:`L_2` regularization strength
                of the Bayesian Personalized Ranking (BPR) loss.
                (default: :obj:`1e-4`)
            **kwargs (optional): Additional arguments of the underlying
                :class:`torch_geometric.nn.models.lightgcn.BPRLoss` loss
                function.
        """
        loss_fn = BPRLoss(lambda_reg, **kwargs)
        emb = self.emb.weight
        emb = emb if node_id is None else emb[node_id]
        return loss_fn(pos_edge_rank, neg_edge_rank, emb)

    def forward(
        self,
        edge_index: Adj,
        edge_label_index: OptTensor = None,
        edge_weight: OptTensor = None,
    ) -> Tensor:
        r"""Computes rankings for pairs of nodes.

        Args:
            edge_index (torch.Tensor or SparseTensor): Edge tensor specifying
                the connectivity of the graph after the GCN normalization.
            edge_label_index (torch.Tensor, optional): Edge tensor specifying
                the node pairs for which to compute rankings or probabilities.
                If :obj:`edge_label_index` is set to :obj:`None`, all edges in
                :obj:`edge_index` will be used instead. (default: :obj:`None`)
            edge_weight (torch.Tensor, optional): The weight of each edge in
                :obj:`edge_index` after the GCN normalization.
                (default: :obj:`None`)
        """
        if edge_label_index is None:
            if is_sparse(edge_index):
                edge_label_index, _ = to_edge_index(edge_index)
            else:
                edge_label_index = edge_index
        all_embs = self.get_embedding(edge_index, edge_weight)
        out_src = all_embs[edge_label_index[0]]
        out_dst = all_embs[edge_label_index[1]]
        return (out_src * out_dst).sum(dim=-1)

    def recommend(
        self,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        src_index: OptTensor = None,
        dst_index: OptTensor = None,
        k: int = 1,
        sorted: bool = True,
    ) -> Tensor:
        r"""Get top-:math:`k` recommendations for nodes in :obj:`src_index`.

        Args:
            edge_index (torch.Tensor or SparseTensor): Edge tensor specifying
                the connectivity of the graph after the GCN normalization.
            edge_weight (torch.Tensor, optional): The weight of each edge in
                :obj:`edge_index` after the GCN normalization.
                (default: :obj:`None`)
            src_index (torch.Tensor, optional): Node indices for which
                recommendations should be generated.
                If set to :obj:`None`, all nodes will be used.
                (default: :obj:`None`)
            dst_index (torch.Tensor, optional): Node indices which represent
                the possible recommendation choices.
                If set to :obj:`None`, all nodes will be used.
                (default: :obj:`None`)
            k (int, optional): Number of recommendations. (default: :obj:`1`)
            sorted (bool, optional): Whether to sort the recommendations
                by score. (default: :obj:`True`)
        """
        out_src = out_dst = self.get_embedding(edge_index, edge_weight)

        if src_index is not None:
            out_src = out_src[src_index]

        if dst_index is not None:
            out_dst = out_dst[dst_index]

        pred = out_src @ out_dst.t()
        top_index = pred.topk(k, dim=-1, sorted=sorted).indices

        if dst_index is not None:  # Map local top-indices to original indices.
            top_index = dst_index[top_index.view(-1)].view(*top_index.size())
        return top_index

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.num_nodes}, '
                f'{self.emb_size}, num_layers={self.num_layers})')
