import torch
import torch.nn.functional as F
from torch import nn

from torch_geometric.data import Data
from torch_geometric.typing import Adj, Tensor
from torch_geometric.utils import is_sparse, scatter


def pprgo_prune_features(data: Data) -> Data:
    r"""Prunes the node features in :obj:`data` so that the only vectors
    loaded into memory correspond to target nodes, as prescribed by
    :obj:`data.edge_index`. Useful for saving memory during PPRGo training.

    Args:
        data (Data): Graph object.

    :rtype: :class:`Data`
    """
    data.x = data.x[data.edge_index[1], :]
    return data


class PPRGo(nn.Module):
    r"""The PPRGo model, based on efficient propagation of approximate
    personalized vectors from the `"Scaling Graph Neural Networks with
    Approximate PageRank" <https://arxiv.org/pdf/2007.01570>`_ paper.

    Propagates pointwise predictions on node embeddings according to truncated
    sparse PageRank vectors. Because this model considers all :math:`K`-hop
    neighborhoods simultaneously, it is only one layer and fast.

    Prior to training, a sparse approximation of PPR vectors should be
    efficiently precomputed via :class:`torch_geometric.transforms.gdc.GDC`.
    This information is expected as :obj:`edge_index` and :obj:`edge_attr`
    during the forward pass.

    Args:
        num_features (int): Number of dimensions in node features.
        num_classes (int): Number of output classes.
        hidden_size (int): Number of hidden dimensions.
        n_layers (int): Number of linear layers for pointwise node projections.
            Minimum 1.
        dropout (float, optional): Node dropout probability for diffused graph.
            (default: :obj:`0.0`)
        **kwargs (optional): Additional arguments to
            :class:`torch.nn.Module`.
    """
    def __init__(self, num_features: int, num_classes: int, hidden_size: int,
                 n_layers: int, dropout: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Initialize MLP for feature computation, our only parameters
        assert n_layers >= 1, "Must have at least 1 layer to ensure dims work"
        if n_layers == 1:
            fcs = [nn.Linear(num_features, num_classes, bias=False)]
        else:
            fcs = [nn.Linear(num_features, hidden_size, bias=False)]
            for i in range(n_layers - 2):
                fcs.append(nn.Linear(hidden_size, hidden_size, bias=False))
            fcs.append(nn.Linear(hidden_size, num_classes, bias=False))

        self.fcs = nn.ModuleList(fcs)
        self.drop = nn.Dropout(dropout)

    def _features(self, x: Tensor) -> Tensor:
        r"""Compute MLP features on :obj:`x`.

        :rtype: :class:`Tensor`
        """
        x = self.fcs[0](self.drop(x))
        for fc in self.fcs[1:]:
            x = fc(self.drop(F.relu(x)))
        return x

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor) -> Tensor:
        r"""Computes the forward pass through PPRGo, assuming PPR edges
        and scores are precomputed and accessible via :obj:`edge_index`
        and :obj:`edge_attr`. Note this expects a truncated node feature
        matrix, which can be produced by calling
        :obj:`torch_geometric.nn.models.pprgo.pprgo_prune_features` on the
        train graph.

        Args:
            x (Tensor): Truncated node feature matrix with shape
                :obj:`[|V|, d]`. :obj:`|V|` is the number of edges.
                Each row corresponds to an embedding of a destination node.
            edge_index (Adj): :obj:`[2, |V|]` dense matrix of PPR edges.
            edge_attr (Tensor): :obj:`[|V|,]` vector of PPR scores.

        :rtype: :class:`Tensor`
        """
        if is_sparse(edge_index):  # pragma: no cover
            # We expect only the [2, |E|] dense format for now
            raise ValueError("Sparse tensors not supported yet")

        # First perform node feature computation (compute logits)
        x = self._features(x)

        # Next manually scatter along the correct edges via PPR
        # It would be nice to use the MessagePassing base class here,
        # but because of the safety checks and the fact that x is
        # pre-indexed during data loading, this is not possible
        weighted = x * edge_attr[:, None]
        src_idx = edge_index[0]
        dim_size = src_idx[-1] + 1

        # We expect a src_idx with every node as a source node (ordered)
        # since topk threshold will leave k>0 outgoing edges per node
        return scatter(weighted, src_idx, dim=0, dim_size=dim_size,
                       reduce='sum')

    @torch.no_grad()
    def predict_power_iter(self, x: Tensor, edge_index: Adj,
                           n_power_iters: int = 1, alpha: float = 0.15,
                           frac_predict: float = 1.0,
                           normalization: str = "sym",
                           batch_size: int = 8192) -> Tensor:
        r"""Forward pass through PPRGo with power iteration instead of
        computing all the sparse PPR vectors. Useful for large graphs.

        During inference, we only need to compute a small set of node
        predictions. These labels are then propagated across the rest of
        the nodes, assuming graph homophily. This propagation is expressed as

        .. math::
            \mathbf{Q}^{(0)} &= \mathbf{H}

            \mathbf{Q}^{(p+1)} &= (1-\alpha) \mathbf{D}^{-1} \mathbf{AQ}^{(p)}
            + \alpha \mathbf{H}

        where :math:`\mathbf{H}` are the predictions on the reduced set.

        Since we might need inference on a large graph, we batch prediction
        computations via :obj:`batch_size`. The returned logits are shape
        :obj:`[|V|, num_classes]`.

        Args:
            x (Tensor): Node features.
            edge_index (Tensor): Adjacency matrix with shape :obj:`[2, |E|]`.
            n_power_iters (int, optional): Number of power iterations.
                (default: :obj:`1`)
            alpha (float, optional): Teleportation probability.
                (default: :obj:`0.15`)
            frac_predict (float, optional): Fraction of nodes to run feature
                computation for. All other features will be diffused during
                message propagation but initially set to 0.
                (default: :obj:`1.0`)
            normalization (str, optional): Determines normalization of
                :math:`\mathbf{A}` during power iteration. Should match the
                :obj:`in_normalization` in
                :obj:`torch_geometric.transforms.gdc.GDC`.
                For now, only :obj:`'sym'` normalization is supported.
                (default: :obj:`'sym'`)
            batch_size (int, optional): Batch size for computing predictions
                before label propagation.
                (default: :obj:`8192`)

        :rtype: :class:`Tensor`
        """
        assert n_power_iters >= 1, "Number of iterations must be positive int"

        # First, sample node embeddings along edges according to frac_predict
        n_nodes = x.shape[0]
        if frac_predict != 1.0:
            ind = torch.randperm(n_nodes)[:int(frac_predict * n_nodes)]
            ind, _ = torch.sort(ind)
            x = x[ind]
        else:
            ind = torch.arange(0, n_nodes)

        # Then, compute logits on the selected nodes (on gpu if possible)
        # propagating to non-selected nodes as well (assumes graph homophily)
        # Since even the number of selected nodes may be large, batch inference
        device = next(self.fcs.parameters()).device
        train = self.fcs.training
        if train:
            self.fcs.eval()

        sele_logits = []
        for j in range(0, n_nodes, batch_size):
            x_batch = x[j:j + batch_size].to(device)
            preds = self._features(x_batch).cpu()
            sele_logits.append(preds)

        sele_logits = torch.vstack(sele_logits)
        if train:
            self.fcs.train()

        # Set all other logits to zero in the graph, they will get
        # filled in when we propagate the selected nodes
        logits_init = torch.zeros((n_nodes, sele_logits.shape[1]),
                                  dtype=torch.float32)
        logits_init[ind] = sele_logits.to(torch.float32)

        # Finally, run power iteration (differ based on normalization)
        try:
            from torch_sparse import SparseTensor
        except ImportError:  # pragma: no cover
            raise ValueError(
                "Cannot find torch_sparse package, needed for inference")

        logits = logits_init.clone()
        adj = SparseTensor(row=edge_index[0], col=edge_index[1],
                           sparse_sizes=(n_nodes, n_nodes))

        if normalization == 'sym':
            # Assume undirected (symmetric) adjacency matrix
            # (In practice, topk sparsification usually leads to some rounding
            # errors which slightly violate this symmetry)
            denom = torch.maximum(adj.sum(1).flatten(), torch.Tensor([1e-12]))
            deg_sqrt_inv = torch.unsqueeze(1. / torch.sqrt(denom), dim=1)
            for j in range(n_power_iters):
                deg_adj_logits = adj @ (deg_sqrt_inv * logits)
                logits = ((1 - alpha) * deg_sqrt_inv * deg_adj_logits +
                          alpha * logits_init)
        else:  # pragma: no cover
            raise NotImplementedError(normalization + " norm not implemented")

        return logits
