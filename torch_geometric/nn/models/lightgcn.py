from torch_geometric.typing import Adj

import torch
from torch import Tensor, LongTensor
import torch.nn.functional as F
from torch.nn import ModuleList, Embedding, BCELoss
from torch.nn.modules.loss import _Loss

from torch_geometric.nn.conv import LGConv


class LightGCN(torch.nn.Module):
    r"""The LightGCN model from the `"LightGCN: Simplifying and Powering
    Graph Convolution Network for Recommendation"
    <https://arxiv.org/abs/2002.02126>`_ paper.

    LightGCN learns embeddings by linearly propagating them on the underlying
    graph, and uses the weighted sum of the embeddings learned at all layers as
    the final embedding as follows:

    .. math::
        \textbf{x}_i = \sum_{l=0}^{L} \alpha_l \textbf{x}^{l}_i

    where each layer's embedding is computed as:

    .. math::
        \mathbf{x}^{l+1}_i = \sum_{j \in \mathcal{N}(i)}
        \frac{1}{\sqrt{\deg(i)}\sqrt{\deg(j)}}\mathbf{x}^{l}_j

    Two prediction heads are provided: :obj:`"ranking"` and
    :obj:`"link_prediction"`. Ranking is performed by outputting the dot
    product between the pair of embeddings chosen for evaluation. For link
    prediction, the sigmoid of the ranking is computed to output label
    probabilities.

    .. note::

        This implementation works not only on bipartite graphs but on any
        kind of connected graph. Embeddings are propagated according to the
        graph connectivity specified by :obj:`edge_index` while rankings or
        probabilities are only computed (the forward pass) according to the
        edges specified by :obj:`edge_label_index`.

    Args:
        num_layers (int): The number of
            :class:`~torch_geometric.nn.conv.LGConv` layers.
        num_nodes (int): The number of nodes in the graph.
        embedding_dim (int, optional): The dimensionality of the node
            embeddings (default: 32).
        objective (str, optional): The objective of the model. The original
            paper implements the :obj:`"ranking"` objective with a simple dot
            product between embeddings as a prediction head. The option
            :obj:`"link_prediction"` passes the ranking result through a
            sigmoid layer (:obj:`"ranking"`, :obj:`"link_prediction"`)
            (default: :obj:`"ranking"`).
        alpha (Tensor, optional): The vector specifying the layer weights in
            the final embeddings. If set to :obj:`None`, the original paper's
            uniform initialization of :obj:`(1 / num_layers + 1)` is used
            (default: :obj:`None`).
        **kwargs (optional): Additional arguments of the underlying
            :class:`torch_geometric.nn.conv.LGConv` layers.
    """

    objectives = ['ranking', 'link_prediction']

    def __init__(self, num_layers: int, num_nodes: int,
                 embedding_dim: int = 32, objective: str = 'ranking',
                 alpha: Tensor = None, **kwargs):
        super().__init__()
        assert (objective in LightGCN.objectives), 'Objective must be one ' \
            f'of {LightGCN.objectives}'

        if isinstance(alpha, Tensor):
            assert (alpha.size(0) == (num_layers + 1)), 'alpha must be of ' \
                'size (num_layers + 1).'
        else:
            alpha = Tensor([1 / (num_layers + 1)] * (num_layers + 1))
            self.register_buffer('alpha', alpha)

        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.embeddings = Embedding(num_nodes, embedding_dim)
        self.objective = objective
        self.alpha = alpha
        self.convs = ModuleList()

        for _ in range(num_layers):
            self.convs.append(LGConv(**kwargs))

    def reset_parameters(self):
        self.embeddings.reset_parameters()

    def forward(self, edge_index: Adj, edge_label_index: Adj = None) -> Tensor:
        """Computes rankings or link probabilities for pairs of nodes.

        Args:
            edge_index (Adj): Edge tensor specifying the connectivity of the
                graph.
            edge_label_index (Adj, optional): Edge tensor specifying the start
                and end nodes for which to compute rankings or proabilities.
                If :obj:`edge_label_index` is set to :obj:`None`, all edges in
                :obj:`edge_index` will be used (default: :obj:`None`).
        """
        if edge_label_index is None:
            edge_label_index = edge_index

        out = self._propagate_embeddings(edge_index)

        out_src = out[edge_label_index[0]]
        out_dst = out[edge_label_index[1]]
        out = torch.sum(out_src * out_dst, dim=-1)

        if self.objective == LightGCN.objectives[1]:
            out = torch.sigmoid(out)

        return out

    def predict_link(self, edge_index: Adj, edge_label_index: Adj = None,
                     prob: bool = False) -> Tensor:
        """Predict links between nodes specified in :obj:`edge_label_index`.

        This prediction head works only for the "link_prediction" objective.

        Args:
            edge_index (Adj): Edge tensor specifying the connectivity of the
                graph.
            edge_label_index (Adj, optional): Edge tensor specifying the start
                and end nodes for which to compute rankings or proabilities.
                If :obj:`edge_label_index` is set to :obj:`None`, all edges in
                :obj:`edge_index` will be used (default: :obj:`None`).
            prob (bool): Whether probabilities should be returned.
        """
        assert (self.objective == LightGCN.objectives[1]), 'Prediction ' \
            'only works for "link_prediction" objective.'

        if edge_label_index is None:
            edge_label_index = edge_index

        predictions = self(edge_index, edge_label_index)
        if prob:
            return predictions

        return torch.round(predictions)

    def recommend(self, edge_index: Adj, source_indices: LongTensor,
                  topK: int = 1, target_indices: LongTensor = None) -> Tensor:
        """Get :obj:`topK` recommendations for nodes in :obj:`source_indices`.

        This prediction head works only for the "ranking" objective.

        Args:
            edge_index (Adj): Edge tensor specifying the connectivity of the
                graph.
            source_indices (LongTensor): Nodes for which recommendations
                should be generated (indices).
            topK (int, optional): Number of top recommendations to return
                (default: 1).
            target_indices (LongTensor, optional): Nodes which represent the
                possible recommendation choices. If set to :obj:`None`, all
                nodes are possible recommendation choices (default:
                :obj:`None`).
        """
        assert (self.objective == LightGCN.objectives[0]), 'Recommendations ' \
            'only work for "ranking" objective.'

        prop_embeddings = self._propagate_embeddings(edge_index)
        source_embeddings = prop_embeddings[source_indices]

        target_embeddings = prop_embeddings
        if isinstance(target_indices, LongTensor):
            target_embeddings = prop_embeddings[target_indices]

        rankings = torch.mm(source_embeddings, target_embeddings.t())
        top_indices = torch.topk(rankings, topK, dim=-1).indices

        if not isinstance(target_indices, LongTensor):
            return top_indices

        index_matrix = target_indices.reshape((1, -1))
        index_matrix = index_matrix.repeat(source_indices.size(0), 1)

        return index_matrix.gather(-1, top_indices)

    def link_prediction_loss(self, predictions: Tensor,
                             edge_label: Tensor, **kwargs) -> Tensor:
        """Computes model loss for the :obj:`"link_prediction"` objective.

        Computes the Binary Cross-Entropy (BCE) loss for link prediction tasks.

        Args:
            preds (Tensor): Predictions for link prediction objective.
            edge_label (Tensor): Edge label for link prediction.
            **kwargs (optional): Additional arguments of the underlying
                :class:`torch.nn.BCELoss` loss function.
        """
        loss_fn = BCELoss()
        return loss_fn(predictions, edge_label.to(torch.float32), **kwargs)

    def recommendation_loss(self, pos_edge_ranks: Tensor,
                            neg_edge_ranks: Tensor, lambda_reg: float = 1e-4,
                            **kwargs) -> Tensor:
        """Computes model loss for the :obj:`"ranking"` objective.

        Computes the Bayesian Personalized Ranking (BPR) loss ranking tasks.

        .. note::

            The i-th entry in the :obj:`pos_edge_ranks` vector and i-th entry
            in the :obj:`neg_edge_ranks` entry must correspond to ranks of
            positive and negative edges of the same entity (i.e. user).

        Args:
            pos_edge_ranks (Tensor): Positive edge rankings for ranking
                objective (default: :obj:`None`).
            neg_edge_ranks (Tensor): Negative edge rankings for ranking
                objective (default: :obj:`None`).
            lambda_reg (int, optional): The :math:`L_2` regularization strength
                of the Bayesian Personalized Ranking (BPR) loss
                (default: 1e-4).
            **kwargs (optional): Additional arguments of the underlying
                :class:`torch_geometric.nn.models.lightgcn.BPRLoss` loss
                function.
        """
        loss_fn = BPRLoss(lambda_reg, **kwargs)
        return loss_fn(pos_edge_ranks, neg_edge_ranks, self.embeddings.weight)

    def get_embeddings(self, edge_index: Adj,
                       indices: LongTensor = None) -> Tensor:
        """"""
        if indices is None:
            return self._propagate_embeddings(edge_index)

        return self._propagate_embeddings(edge_index)[indices]

    def _propagate_embeddings(self, edge_index: Adj) -> Tensor:
        x = self.embeddings.weight
        out = x * self.alpha[0]

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            out = out + x * self.alpha[i + 1]

        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'num_layers={self.num_layers}, '
                f'num_nodes={self.num_nodes}, '
                f'embedding_dim={self.embedding_dim}, '
                f'objective=\"{self.objective}\")')


class BPRLoss(_Loss):
    r"""Bayesian Personalized Ranking (BPR) loss

    The Bayesian Personalized Ranking (BPR) loss is a pairwise loss that
    encourages the prediction of an observed entry to be higher than its
    unobserved counterparts (see the paper `"LightGCN: Simplifying and Powering
    Graph Convolution Network for Recommendation"
    <https://arxiv.org/abs/2002.02126>`_).

    .. math::
        L_{\text{BPR}} = - \sum_{u=1}^{M} \sum_{i \in \mathcal{N}_u}
        \sum_{j \not\in \mathcal{N}_u} \ln \sigma(\hat{y}_{ui} - \hat{y}_{uj})
        + \lambda \vert\vert \textbf{x}^{(0)} \vert\vert^2

    where :math:`lambda` controls the :math:`L_2` regularization strength. We
    compute the mean BPR loss for simplicity.

    Args:
        lambda_reg (float, optional): The :math:`L_2` regularization strength
            (default: 0).
        **kwargs (optional): Additional arguments of the underlying
            :class:`torch.nn.modules.loss._Loss` class.
    """
    __constants__ = ['lambda_reg']
    lambda_reg: float

    def __init__(self, lambda_reg: float = 0, **kwargs) -> None:
        super(BPRLoss, self).__init__(None, None, "sum", **kwargs)
        self.lambda_reg = lambda_reg

    def forward(self, positives: Tensor, negatives: Tensor,
                parameters: Tensor = None) -> Tensor:
        """Compute the mean Bayesian Personalized Ranking (BPR) loss.

        .. note::

            The i-th entry in the :obj:`positives` vector and i-th entry
            in the :obj:`negatives` entry should correspond to the same
            entity (i.e. user), as the BPR is a personalized ranking loss.

        Args:
            positives (Tensor): The vector of positive-pair rankings.
            negatives (Tensor): The vector of negative-pair rankings.
            parameters (Tensor, optional): The tensor of parameters which
                should be used for :math:`L_2` regularization
                (default: :obj:`None`).
        """
        n_pairs = positives.size(0)
        log_prob = F.logsigmoid(positives - negatives).mean()
        regularization = 0

        if self.lambda_reg != 0:
            regularization = self.lambda_reg * parameters.norm(p=2).pow(2)

        return (-log_prob + regularization) / n_pairs
