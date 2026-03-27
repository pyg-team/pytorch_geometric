import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.conv import GCNConv, GraphConv
from torch_geometric.nn.pool.select import SelectTopK
from torch_geometric.utils import cumsum, degree, dense_to_sparse, scatter


class SelectSageModule(torch.nn.Module):
    r"""Node selection module based on a GNN scoring function (SAGPool-style).

    Uses a :class:`~torch_geometric.nn.conv.GraphConv` to compute per-node
    scores, then applies a top-:math:`k` selection.

    Args:
        in_channels (int): Size of each input node feature vector.
        ratio (float or int): Ratio (or number) of nodes to keep. If
            :obj:`min_score` is set, this argument is ignored.
            (default: :obj:`0.5`)
        GNN (torch.nn.Module): The GNN operator used to compute node
            scores. (default: :class:`~torch_geometric.nn.conv.GraphConv`)
        min_score (float, optional): Minimal node score threshold.
            If set, nodes whose score is below this threshold are removed.
            (default: :obj:`None`)
        act (str): Activation function applied to scores.
            (default: :obj:`"tanh"`)
    """

    def __init__(
        self,
        in_channels: int,
        ratio: float = 0.5,
        GNN=GraphConv,
        min_score=None,
        act: str = "tanh",
    ):
        super().__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.min_score = min_score

        self.gnn = GNN(in_channels, 1)
        self.select = SelectTopK(1, ratio, min_score, act)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.gnn.reset_parameters()
        self.select.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch=None,
        attn=None,
    ):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        attn = x if attn is None else attn
        attn = attn.view(-1, 1) if attn.dim() == 1 else attn
        attn = self.gnn(attn, edge_index)
        return self.select(attn, batch)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.in_channels}, \
            ratio={self.ratio})"


class SelectTopKModule(torch.nn.Module):
    r"""
    Node selection module based on a learned projection vector (TopK-style).

    Args:
        in_channels (int): Size of each input node feature vector.
        ratio (float or int): Ratio (or number) of nodes to keep. If
            :obj:`min_score` is set, this argument is ignored.
            (default: :obj:`0.5`)
        min_score (float, optional): Minimal node score threshold.
            (default: :obj:`None`)
        act (str): Activation function applied to scores.
            (default: :obj:`"tanh"`)
    """

    def __init__(
        self,
        in_channels: int,
        ratio: float = 0.5,
        min_score=None,
        act: str = "tanh",
    ):
        super().__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.min_score = min_score

        self.select = SelectTopK(in_channels, ratio, min_score, act)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.select.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor, batch=None):
        return self.select(x, batch)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.in_channels}, \
            ratio={self.ratio})"


class AssociationModule(torch.nn.Module):
    r"""Computes soft assignment scores between all nodes and the selected
    representative nodes.

    Three association modes are supported:

    - :obj:`"scalar"`: Standard dot-product followed by softmax.
    - :obj:`"cosine"`: Cosine-similarity followed by softmax.
    - :obj:`"attn"`: Scaled dot-product attention with learnable projections.

    Args:
        mode (str): Association mode (:obj:`"scalar"`, :obj:`"cosine"`, or
            :obj:`"attn"`). (default: :obj:`"scalar"`)
        in_channels (int, optional): Size of each input node embedding.
            Required when :obj:`mode="attn"`. (default: :obj:`None`)
        attention_dim (int, optional): Dimension of the attention space.
            Required when :obj:`mode="attn"`. (default: :obj:`None`)
    """

    def __init__(
        self,
        mode: str = "scalar",
        in_channels=None,
        attention_dim=None,
    ):
        super().__init__()

        if mode not in ["scalar", "cosine", "attn"]:
            raise NotImplementedError(
                "Association mode not recognised. "
                "Available modes: 'scalar', 'cosine', 'attn'."
            )

        self.mode = mode

        if mode == "attn":
            if in_channels is None or attention_dim is None:
                raise ValueError(
                    "'in_channels' and 'attention_dim' must be set \
                        when mode='attn'."
                )
            self.attention_dim = attention_dim
            self.Wq = torch.nn.Linear(in_channels, attention_dim, bias=False)
            self.Wk = torch.nn.Linear(in_channels, attention_dim, bias=False)

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        if self.mode == "attn":
            self.Wq.reset_parameters()
            self.Wk.reset_parameters()

    def forward(self, x: Tensor, xrep: Tensor) -> Tensor:
        r"""
        Args:
            x (torch.Tensor): Node embeddings :math:`(N, d)`.
            xrep (torch.Tensor): Representative node embeddings
                :math:`(k, d)`.

        Returns:
            torch.Tensor: Soft assignment matrix :math:`(N, k)`.
        """
        if self.mode == "scalar":
            return F.softmax(torch.matmul(x, xrep.T), dim=-1)

        elif self.mode == "cosine":
            x_norm = F.normalize(x, p=2, dim=1)
            xrep_norm = F.normalize(xrep, p=2, dim=1)
            return F.softmax(torch.matmul(x_norm, xrep_norm.T), dim=-1)

        else:  # attn
            query = self.Wq(x)
            key = self.Wk(xrep)
            scores = torch.matmul(query, key.T) / (self.attention_dim**0.5)
            return F.softmax(scores, dim=-1)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mode={self.mode})"


class SPAPooling(torch.nn.Module):
    r"""
    The sparse pooling operator from the
    `"SpaPool: Soft Partition Assignment Pooling for Graph Neural Networks"
    https://doi.org/10.1007/978-3-032-02215-8_27`_ paper.

    SpaPool performs hierarchical graph pooling by (1) embedding nodes with a
    :class:`~torch_geometric.nn.conv.GCNConv` layer, (2) selecting a subset of
    representative nodes via a configurable selection module (TopK or
    SAGPool), and (3) computing a soft assignment matrix between all nodes and
    the representatives via an association module.  The coarsened node features
    are obtained as :math:`\mathbf{O} = \mathbf{S}^{\top}\mathbf{X}` and the
    pooled adjacency as
    :math:`\mathbf{A}^{\prime} = \mathbf{S}^{\top}\mathbf{A}\mathbf{S}`.

    Several regularisation losses are supported: *DiffPool*
    (:obj:`"diffpool"`), *MinCut* (:obj:`"mincut"`), *DMoN* (:obj:`"dmon"`),
    or a fully custom list (:obj:`"manual"`).

    Args:
        ratio (float or int): Ratio (or number) of nodes to keep after
            pooling.
        n_feature (int): Number of input node features.
        min_score (float, optional): Minimal node score :math:`\tilde\alpha`.
            If set, :obj:`ratio` is ignored and nodes with score below the
            threshold are dropped. (default: :obj:`None`)
        select (str): Node selection strategy: :obj:`"topk"` or
            :obj:`"sagpool"`. (default: :obj:`"topk"`)
        asso (str): Association mode passed to
            :class:`AssociationModule`: :obj:`"scalar"`, :obj:`"cosine"`, or
            :obj:`"attn"`. (default: :obj:`"attn"`)
        loss (str): Regularisation loss family: :obj:`"diffpool"`,
            :obj:`"mincut"`, :obj:`"dmon"`, or :obj:`"manual"`.
            (default: :obj:`"diffpool"`)
        regs (list[str]): Explicit list of regularisation terms used when
            :obj:`loss="manual"`. Supported values: :obj:`"link"`,
            :obj:`"entropy"`, :obj:`"mincut"`, :obj:`"ortho"`,
            :obj:`"collapse"`, :obj:`"spectral"`, :obj:`"covar"`.
            (default: :obj:`[]`)
        attention_dim (int): Dimension of the attention space used when
            :obj:`asso="attn"`. (default: :obj:`16`)
        no (int): Output dimension of the internal GCN embedding.
            (default: :obj:`5`)

    Shapes:
        - **input:**
          node features :math:`(N, F)`,
          batch vector :math:`(N,)`,
          edge indices :math:`(2, E)`
        - **output:**
          pooled node features :math:`(K, F)`,
          pooled edge indices :math:`(2, E')`,
          :obj:`None` (edge weights placeholder),
          pooled batch vector :math:`(K,)`,
          selected node indices :math:`(K,)`,
          selection scores :math:`(K,)`,
          regularisation loss dict

    Example:
        >>> pool = SpaPool(ratio=0.5, n_feature=16)
        >>> x = torch.randn(10, 16)
        >>> edge_index = torch.randint(0, 10, (2, 30))
        >>> batch = torch.zeros(10, dtype=torch.long)
        >>> out, ei, _, b, perm, score, loss = pool(x, batch, edge_index)
    """

    def __init__(
        self,
        ratio: float,
        n_feature: int,
        min_score=None,
        select: str = "topk",
        asso: str = "attn",
        loss: str = "diffpool",
        regs=None,
        attention_dim: int = 16,
        no: int = 5,
    ):
        super().__init__()

        self.ratio = ratio
        self.n_feature = n_feature
        self.min_score = min_score
        self.loss_fn = loss
        self.regs = regs if regs is not None else []
        self.no = no

        self.conv = GCNConv(in_channels=n_feature, out_channels=no)
        self.act = torch.nn.Sigmoid()

        if select == "topk":
            self.select = SelectTopKModule(
                in_channels=no, ratio=ratio, min_score=min_score, act="tanh"
            )
        elif select == "sagpool":
            self.select = SelectSageModule(
                in_channels=no, ratio=ratio, min_score=min_score, act="tanh"
            )
        else:
            raise NotImplementedError(
                "Only 'topk' or 'sagpool' are supported as select modules."
            )

        self.association = AssociationModule(
            asso, in_channels=no, attention_dim=attention_dim
        )

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.conv.reset_parameters()
        self.association.reset_parameters()
        self.select.reset_parameters()

    def forward(self, x: Tensor, batch: Tensor, edge_index: Tensor):
        r"""Forward pass.

        Args:
            x (torch.Tensor): Node feature matrix :math:`(N, F)`.
            batch (torch.Tensor): Batch vector :math:`(N,)` assigning
                each node to a graph in the batch.
            edge_index (torch.Tensor): Graph connectivity in COO format
                :math:`(2, E)`.

        Returns:
            (torch.Tensor, torch.Tensor, None, torch.Tensor, torch.Tensor,
            torch.Tensor, dict): Tuple of pooled node features, pooled edge
            index, :obj:`None` (placeholder), pooled batch vector, selected
            node permutation, selection scores, and a dict of regularisation
            losses.
        """
        xemb = self.act(self.conv(x, edge_index))  # [N, no]

        # ── Select representatives ──────────────────────────────────────────
        select_out = self.select(x=xemb, edge_index=edge_index, batch=batch)
        perm = select_out.node_index
        score = select_out.weight

        xrep = xemb[perm] * score.view(-1, 1)  # [K, no]

        # ── Per-graph node counts ────────────────────────────────────────────
        n_nodes = scatter(batch.new_ones(x.size(0)), batch, reduce="sum")
        if self.min_score is None:
            k_nodes = (
                        float(self.ratio) * n_nodes.to(x.dtype)
                    ).ceil().to(torch.long)
        else:
            k_nodes = scatter(
                batch[perm].new_ones(xrep.size(0)), batch[perm], reduce="sum"
            )

        cumn = cumsum(n_nodes)
        cumk = cumsum(k_nodes)

        # ── Soft assignment (mask across batches) ───────────────────────────
        attn_scores = self.association(xemb, xrep)
        mask = batch.unsqueeze(1) == batch[perm].unsqueeze(0)
        attn_scores = attn_scores.masked_fill(~mask, float("-inf"))
        S = F.softmax(attn_scores, dim=-1)  # [N, K]

        # ── Pooled features and adjacency ────────────────────────────────────
        out = torch.matmul(S.T, x)  # [K, F]

        adj = torch.zeros(x.size(0), x.size(0), device=x.device)
        adj[edge_index[0], edge_index[1]] = 1.0
        out_adj = torch.matmul(torch.matmul(S.T, adj), S)
        edge_index_pool = dense_to_sparse(out_adj)[0]

        # ── Regularisation losses ────────────────────────────────────────────
        _loss_map = {
            "diffpool": ["link", "entropy"],
            "mincut": ["mincut", "ortho"],
            "dmon": ["ortho", "collapse", "spectral"],
            "manual": self.regs,
        }
        regs = _loss_map[self.loss_fn]

        loss = {}
        for reg in regs:
            if reg == "link":
                loss[reg] = self._link_loss(adj, S)
            elif reg == "entropy":
                loss[reg] = self._entropy_loss(S)
            elif reg == "mincut":
                loss[reg] = self._mincut_loss(x, edge_index, S, out_adj)
            elif reg == "ortho":
                loss[reg] = self._ortho_loss(S, n_nodes, cumn, k_nodes, cumk)
            elif reg == "collapse":
                loss[reg] = self._collapse_loss(
                    S,
                    n_nodes,
                    cumn,
                    k_nodes,
                    cumk,
                )
            elif reg == "spectral":
                loss[reg] = self._spectr_loss(edge_index, S, n_nodes)
            elif reg == "covar":
                loss[reg] = self._covariance_loss(xrep, k_nodes, cumk)

        return out, edge_index_pool, None, batch[perm], perm, score, loss

    # ── Regularisation losses ────────────────────────────────────────────────

    def _link_loss(self, adj: Tensor, S: Tensor) -> Tensor:
        r"""Link prediction regularisation loss."""
        return torch.norm(adj - torch.matmul(S, S.T)) / adj.numel()

    def _entropy_loss(self, S: Tensor) -> Tensor:
        r"""Entropy regularisation loss."""
        return (-S * torch.log(S + 1e-15)).sum(dim=-1).mean()

    def _mincut_loss(
        self, x: Tensor, edge_index: Tensor, S: Tensor, out_adj: Tensor
    ) -> Tensor:
        r"""MinCut regularisation loss (range :math:`[-1, 0]`)."""
        d = torch.diag(degree(edge_index[0], x.size(0)))
        mincut_den = torch.trace(torch.matmul(torch.matmul(S.T, d), S))
        return -torch.trace(out_adj) / mincut_den

    def _ortho_loss(
        self,
        S: Tensor,
        n_nodes: Tensor,
        cumn: Tensor,
        k_nodes: Tensor,
        cumk: Tensor,
    ) -> Tensor:
        r"""Orthogonality regularisation loss (range :math:`[0, 2]`)."""
        loss = S.new_zeros(1)
        for i in range(len(n_nodes)):
            si = S[cumn[i]:cumn[i] + n_nodes[i], cumk[i]:cumk[i] + k_nodes[i]]
            sts = torch.matmul(si.T, si)
            i_s = torch.eye(k_nodes[i], device=S.device, dtype=S.dtype) / (
                k_nodes[i] ** 0.5
            )
            loss = loss + torch.norm(sts / torch.norm(sts) - i_s)
        return loss.squeeze(0) / len(k_nodes)

    def _collapse_loss(
        self,
        S: Tensor,
        n_nodes: Tensor,
        cumn: Tensor,
        k_nodes: Tensor,
        cumk: Tensor,
    ) -> Tensor:
        r"""Collapse regularisation loss."""
        loss = S.new_zeros(1)
        for i in range(len(n_nodes)):
            si = S[cumn[i]:cumn[i]+n_nodes[i],
                   cumk[i]:cumk[i]+k_nodes[i]]
            loss += (
                    torch.norm(si.sum(0)) * (k_nodes[i] ** 0.5)
                     ) / n_nodes[i] - 1
        return loss / len(n_nodes)

    def _spectr_loss(
        self,
        edge_index: Tensor,
        S: Tensor,
        n_nodes: Tensor,
    ) -> Tensor:
        r"""Spectral (modularity) regularisation loss."""
        N = n_nodes.sum()
        d = degree(edge_index[0])
        m = 0.5 * d.sum()
        adj_dense = torch.zeros(N, N, device=S.device)
        adj_dense[edge_index[0], edge_index[1]] = 1.0
        B = adj_dense - d.unsqueeze(-1) * d / (2.0 * m)
        return -torch.trace(torch.matmul(torch.matmul(S.T, B), S)) / (2.0 * m)

    def _covariance_loss(
        self,
        xrep: Tensor,
        k_nodes: Tensor,
        cumk: Tensor,
    ) -> Tensor:
        r"""Covariance regularisation loss."""
        loss = xrep.new_zeros(1)
        for i in range(len(k_nodes)):
            xri = xrep[cumk[i]:cumk[i] + k_nodes[i]]
            xc = xri - xri.mean(dim=0, keepdim=True)
            cov = torch.matmul(xc.T, xc) / (xc.size(0) - 1)
            loss = loss + cov.pow(2).sum() - cov.diag().pow(2).sum()
        return loss / len(k_nodes)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n_feature={self.n_feature}, "
            f"ratio={self.ratio}, "
            f"loss={self.loss_fn})"
        )
