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

    Each layer's embeddings are computed as:

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
        num_layers (int): The number of Light Graph Convolution
            (:obj:`LGConv`) layers.
        num_nodes (int): The number of nodes in the graph.
        embedding_dim (int, optional): The dimensionality of the node
            embeddings (default: 32).
        objective (str, optional): The objective of the model. The original
            paper implements the :obj:`"ranking"` objective with a simple dot
            product between embeddings as a prediction head. The option
            :obj:`"link_prediction"` passes the ranking result through a
            sigmoid layer (:obj:`"ranking"`, :obj:`"link_prediction"`)
            (default: :obj:`"ranking"`).
        lambda_reg (int, optional): The :math:`L_2` regularization strength of
            the Bayesian Personalized Ranking (BPR) loss for the "ranking"
            objective (default: 1e-4).
        alpha (float, optional): The vector specifying the layer weights in the
            final embeddings. If set to :obj:`None`, the original paper's
            uniform initialization of :obj:`(1 / num_layers + 1)` is used
            (default: :obj:`None`).
        **kwargs (optional): Additional arguments of the underlying
            :class:`torch_geometric.nn.conv.MessagePassing` layers.
    """

    objectives = ['ranking', 'link_prediction']

    def __init__(self, num_layers: int, num_nodes: int,
                 embedding_dim: int = 32, objective: str = 'ranking',
                 lambda_reg: float = 1e-4, alpha: Tensor = None, **kwargs):
        super().__init__(**kwargs)

        assert (num_layers >= 1), 'Number of layers must be larger than 0.'
        assert (num_nodes >= 1), 'Number of nodes must be larger than 0.'
        assert (embedding_dim >= 1), 'Embedding dimension must be larger ' \
            'than 0.'
        assert (objective in LightGCN.objectives), 'Objective must be one ' \
            f'of {LightGCN.objectives}'

        if alpha is not None:
            assert (alpha.size(0) == (num_layers + 1)), 'alpha must be of ' \
                'size (num_layers + 1).'
        else:
            alpha = [1 / (num_layers + 1)] * (num_layers + 1)

        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.embeddings = Embedding(num_nodes, embedding_dim)
        self.objective = objective
        self.lambda_reg = lambda_reg
        self.alpha = alpha
        self.convs = ModuleList()

        for _ in range(num_layers):
            self.convs.append(LGConv())

        if objective == LightGCN.objectives[0]:
            self.loss_fn = BPRLoss(self.lambda_reg)
        else:
            self.loss_fn = BCELoss()

    def reset_parameters(self):
        self.embeddings.reset_parameters()

    def forward(self, edge_index: Adj, edge_label_index: Adj = None,
                weights: Tensor = None, **kwargs) -> Tensor:
        """Computes rankings or link probabilities for pairs of nodes.

        Args:
            edge_index (Adj): Edge tensor specifying the connectivity of the
                graph.
            edge_label_index (Adj, optional): Edge tensor specifying the start
                and end nodes for which to compute rankings or proabilities.
                If :obj:`edge_label_index` is set to :obj:`None`, all edges in
                :obj:`edge_index` will be used (default: :obj:`None`).
            weights (Tensor, optional): Pre-computed embeddings or node
                features which should be used to initialize the embedding
                layer (default: :obj:`None`).
        """
        if edge_label_index is None:
            edge_label_index = edge_index

        if weights is not None:
            assert (weights.shape == self.embeddings.shape), \
                'Pre-computed embeddings weights must match shape of \
                embedding layer.'
            self.embeddings.weight.data.copy_(weights)

        out = self._propagate_embeddings(edge_index)

        nodes_fst = torch.index_select(out, 0, edge_label_index[0, :].long())
        nodes_sec = torch.index_select(out, 0, edge_label_index[1, :].long())
        out = torch.sum(nodes_fst * nodes_sec, dim=-1)

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

        predictions = self.forward(edge_index, edge_label_index)
        if prob:
            return predictions

        return torch.round(predictions)

    def recommend(self, edge_index: Adj, source_indices: LongTensor,
                  target_indices: LongTensor, topK: int = 1) -> Tensor:
        """Get :obj:`topK` recommendations for nodes in :obj:`source_indices`.

        This prediction head works only for the "ranking" objective.

        Args:
            edge_index (Adj): Edge tensor specifying the connectivity of the
                graph.
            source_indices (LongTensor): Nodes for which recommendations
                should be generated (indices).
            target_indices (LongTensor): Nodes which represent the choices
                (indices).
            topK (int, optional): Number of top recommendations to return
                (default: 1).
        """
        assert (self.objective == LightGCN.objectives[0]), 'Recommendations ' \
            'only work for "ranking" objective.'

        prop_embeddings = self._propagate_embeddings(edge_index)
        source_embeddings = prop_embeddings[source_indices]
        target_embeddings = prop_embeddings[target_indices]

        rankings = torch.mm(source_embeddings, target_embeddings.t())
        top_indices = torch.topk(rankings, topK, dim=-1).indices
        index_matrix = target_indices.reshape((1, -1))
        index_matrix = index_matrix.repeat(source_indices.size(0), 1)

        return index_matrix.gather(-1, top_indices)

    def loss(self, preds: Tensor = None, edge_label: Tensor = None,
             pos_ranks: Tensor = None, neg_ranks: Tensor = None
             ) -> Tensor:
        """Computes model loss based on objective.

        Computes Bayesian Personalized Ranking (BPR) loss for ranking objective
        and Binary Cross-Entropy (BCE) loss for link prediction objective.

        .. note::

            If objective is ranking, the i-th entry in the :obj:`pos_ranks`
            vector and i-th entry in the :obj:`neg_ranks` entry must correspond
            to positive and negative edges of the same entity (i.e. user).

        Args:
            preds (Tensor, optional): Predictions for link prediction objective
                (default: :obj:`None`).
            edge_label (Tensor, optional): Edge label for link prediction
                objective (default: :obj:`None`).
            pos_ranks (Tensor, optional): Positive edge rankings for ranking
                objective (default: :obj:`None`).
            neg_ranks (Tensor, optional): Negative edge rankings for ranking
                objective (default: :obj:`None`).
        """
        if self.objective == LightGCN.objectives[0]:
            assert (pos_ranks is not None and neg_ranks is not None), \
                'For ranking objective, pos_ranks and neg_ranks must ' \
                'be specified.'

            return self.loss_fn(pos_ranks, neg_ranks, self.embeddings.weight)

        else:
            assert (preds is not None and edge_label is not None), \
                'For link predictions objective, preds and edge_label must ' \
                'be specified.'

            return self.loss_fn(preds, edge_label.to(torch.float32))

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
            out = torch.add(out, x * self.alpha[i + 1])

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
    """
    __constants__ = ['lambda_reg']
    lambda_reg: float

    def __init__(self, lambda_reg: float = 0) -> None:
        super(BPRLoss, self).__init__(None, None, "sum")
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
