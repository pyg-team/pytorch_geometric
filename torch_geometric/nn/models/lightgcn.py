from torch_geometric.typing import Adj

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import ModuleList, Embedding, BCELoss
from torch.nn.modules.loss import _Loss

from torch_geometric.nn.conv import LGConv


class LightGCN(torch.nn.Module):
    r"""The LightGCN model from the `"LightGCN: Simplifying and Powering
    Graph Convolution Network for Recommendation"
    <https://arxiv.org/abs/2002.02126>`_paper.

    LightGCN learns embeddings by linearly propagating them on the underlying
    graph, and uses the weighted sum of the embeddings learned at all layers as
    the final embedding as follows:

    .. math::
        \textbf{x}_i = \sum_{l=0}^{L} \alpha_l \textbf{x}^{l}_i

    where :math:`\alpha_l` can be chosen to be any weights. The original paper
    uniformly initializes :math:`\alpha_l` as
    :math:`\frac{1}{\text{# layers } + 1}`.

    Each layer's embeddings are computed as:

    .. math::
        \mathbf{x}^{l+1}_i = \sum_{j \in \mathcal{N}(i)}
        \frac{1}{\sqrt{\deg(i)}\sqrt{\deg(j)}}\mathbf{x}^{l}_j

    Two prediction heads are provided (:obj:`"ranking"` and
    :obj:`"link_prediction"`). Ranking is performed by outputting the dot
    product between the pair of embeddings chosen for evaluation. For link
    prediction, the sigmoid of the ranking is computed to output binary
    labels.

    This implementation works not only on bipartite graphs but on any kind of
    connected graph. It is important that all provided edges for evaluation
    in the forward function (:obj:`edge_label_index`) are only edges of the
    kind which need to be predicted (i.e. user-item edges, and not user-user or
    item-item edges).

    Args:
        num_layers (int): The number of Light Graph Convolution :obj:`LGConv`
        layers.
        num_nodes (int): The number of nodes in the graph.
        embedding_dim (int, optional): The dimensionality of the node
            embeddings (default: 32).
        objective (str, optional): The objective of the model. The original
            paper implements the :obj:`"ranking"` objective with a simple dot
            product between embeddings as a prediction head. The option
            :obj:`"link_prediction"` passes the ranking result through a
            sigmoid layer (:obj:`"ranking"`, :obj:`"link_prediction"`)
            (default: :obj:`"ranking"`).
        weights (Tensor, optional): Pre-computed embeddings or node features
            which should be used to initialize the embedding layer. Overwrites
            the :obj:`embedding_dim` argument (default: :obj:`None`).
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
                 weights: Tensor = None, lambda_reg: float = 1e-4,
                 alpha: Tensor = None, **kwargs):
        super().__init__()

        assert (num_layers >= 1), 'Number of layers must be larger than 0.'
        assert (num_nodes >= 1), 'Number of nodes must be larger than 0.'
        assert (embedding_dim >= 1), 'Embedding dimension must be larger \
            than 0.'
        assert (objective in LightGCN.objectives), f'Objective must be one of \
            {LightGCN.objectives}'

        if alpha is not None:
            assert (alpha.size() == (num_layers + 1)), 'Alpha must be valid \
                weights for each layer plus the initial layer summing to one.'
        else:
            alpha = torch.full((1, num_layers + 1), 1 / (num_layers + 1))[0]

        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.objective = objective
        self.lambda_reg = lambda_reg
        self.alpha = alpha
        self.convs = ModuleList()

        if weights is not None:
            self.embeddings = Embedding.from_pretrained(weights)
        else:
            self.embeddings = Embedding(num_nodes, embedding_dim)

        for _ in range(num_layers):
            self.convs.append(LGConv())

        if objective == LightGCN.objectives[0]:
            self.loss_fn = BPRLoss(self.lambda_reg)
        else:
            self.loss_fn = BCELoss()

    def reset_parameters(self):
        self.embeddings.reset_parameters()

    def forward(self, edge_index: Adj, edge_label_index: Tensor,
                **kwargs) -> Tensor:
        """"""
        x = self.embeddings(self.__get_aranged_tensor(edge_index,
                                                      self.num_nodes))
        out = x * self.alpha[0]

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            out = torch.add(out, x * self.alpha[i + 1])

        nodes_fst = torch.index_select(out, 0, edge_label_index[0, :].long())
        nodes_sec = torch.index_select(out, 0, edge_label_index[1, :].long())
        out = torch.sum(nodes_fst * nodes_sec, dim=-1)

        if self.objective == LightGCN.objectives[1]:
            out = torch.sigmoid(out)

        return out

    def recommend(self, source_indices: Tensor, target_indices: Tensor,
                  topK: int = 1):
        """Generate :obj:`topK` recommendations for each node in
           :obj:`source_mask`.

        Args:
            source_indices (Tensor): Nodes for which recommendations should
                be generated (indices).
            target_indices (Tensor): Nodes which represent the choices
                (indices).
            topK (int, optional): Number of top recommendations to return
                (default: 1).
        """
        assert (self.objective == LightGCN.objectives[0]), \
            'Recommendations only work for "ranking" objective. Call \
             predict() instead.'

        source_embeddings = self.embeddings(source_indices)
        target_embeddings = self.embeddings(target_indices)
        rankings = torch.mm(source_embeddings, target_embeddings.t())
        top_indices = torch.topk(rankings, topK, 1).indices.long()
        recs = target_indices.repeat(source_indices.size(0), 1)[top_indices]
        return recs

    def loss(self, edge_label: Tensor, rankings: Tensor = None,
             preds: Tensor = None) -> float:
        """"""
        assert (rankings is not None or preds is not None), \
            'Either rankings or preds must be specified.'

        if rankings is not None:
            indices = self.__get_aranged_tensor(edge_label, self.num_nodes)
            parameters = self.embeddings(indices)
            return self.loss_fn(rankings, edge_label, parameters)

        return self.loss_fn(preds, edge_label)

    def __get_aranged_tensor(self, device_tensor: Tensor, size: int) -> Tensor:
        if device_tensor.is_cuda:
            device = device_tensor.get_device()
            return torch.arange(size, device=f'cuda:{device}')
        else:
            return torch.arange(size)

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

    where :math:`lambda` controls the :math:`L_2` regularization strength.

    Args:
        lambda_reg (float, optional): The :math:`L_2` regularization strength
            (default: 0).
    """
    __constants__ = ['lambda']
    lambda_reg: float

    def __init__(self, lambda_reg: float = 0) -> None:
        super(BPRLoss, self).__init__(None, None, "sum")
        self.lambda_reg = lambda_reg

    def forward(self, rankings: Tensor, link_labels: Tensor,
                parameters: Tensor = None, approximate: bool = True) -> Tensor:
        """

        Args:
            rankings (Tensor): The vector of all computed rankings.
            link_labels (Tensor): The binary vector where ones correspond to
                positive edges or neighbors, respectively, and zeros to
                negative edges or non-neighbors.
            parameters (Tensor, optional): The tensor of parameters which
                should be used for :math:`L_2` regularization
                (default: :obj:`None`).
            approximate (bool, optional): Whether the loss should be computed
                approximately, sampling one contrastive pair per positive
                example. Setting approximate to :obj:`False` performs
                one-to-all contrastive loss which will run out of RAM on
                larger datasets (default: :obj:`True`).
        """
        pos_pairs = torch.masked_select(rankings, link_labels > 0)
        neg_pairs = torch.masked_select(rankings, link_labels < 1)

        log_prob = 0

        if approximate:
            indices = torch.randint_like(pos_pairs, 0,
                                         neg_pairs.size(0)).long()
            neg_samples = neg_pairs[indices]
            log_prob = F.logsigmoid(pos_pairs - neg_samples).mean()

        else:
            for pos_ranking in pos_pairs:
                log_prob += F.logsigmoid(-neg_pairs + pos_ranking).mean()

            log_prob /= pos_pairs.size(0)

        regularization = 0

        if self.lambda_reg != 0:
            regularization = self.lambda_reg * parameters.norm(p=2)

        return -log_prob + regularization
