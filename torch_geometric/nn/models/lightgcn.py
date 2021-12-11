from typing import Optional, Callable, List
from torch_geometric.typing import Adj

import copy

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import ModuleList, Embedding, BCELoss
from torch.nn.modules.loss import _Loss

from torch_geometric.nn.conv import LGConv
from torch_geometric.nn.models.jumping_knowledge import JumpingKnowledge


class LightGCN(torch.nn.Module):
    r"""The continuous-filter convolutional neural network SchNet from the
    `"SchNet: A Continuous-filter Convolutional Neural Network for Modeling
    Quantum Interactions" <https://arxiv.org/abs/1706.08566>`_ paper that uses
    the interactions blocks of the form

    .. math::
        \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \odot
        h_{\mathbf{\Theta}} ( \exp(-\gamma(\mathbf{e}_{j,i} - \mathbf{\mu}))),

    here :math:`h_{\mathbf{\Theta}}` denotes an MLP and
    :math:`\mathbf{e}_{j,i}` denotes the interatomic distances between atoms.

    Args:
        num_layers (int): The number of Light Graph Convolution layers.
        num_nodes (int): The number of nodes in the graph.
        embedding_dim (int, optional): The dimensionality of the node
            embeddings. (default: 32)
        objective (str, optional): The objective of the model. The original
            paper implements the :obj:`"ranking"` objective with a simple dot
            product between embeddings as a prediction head. The option
            :obj:`"link_prediction"` passes the ranking result through a
            sigmoid layer.
            (:obj:`"ranking"`, :obj:`"link_prediction"`).
            (default: :obj:`"ranking"`)
        **kwargs (optional): Additional arguments of the underlying
            :class:`torch_geometric.nn.conv.MessagePassing` layers.
    """

    objectives = ['ranking', 'link_prediction']

    def __init__(self, num_layers: int, num_nodes: int, embedding_dim: int = 32, 
                objective: str = 'ranking', **kwargs):
        super().__init__()
        
        assert (num_layers >= 1), 'Number of layers must be larger than 0.'
        assert (num_nodes >= 1), 'Number of nodes must be larger than 0.'
        assert (embedding_dim >= 1), 'Embedding dimension must be larger than 0.'
        assert (objective in LightGCN.objectives), f'Objective must be one of \
            {LightGCN.objectives}'

        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.objective = objective
        self.embeddings = Embedding(num_nodes, embedding_dim)
        self.convs = ModuleList()

        for _ in range(num_layers):
            self.convs.append(LGConv())
        
        if objective == LightGCN.objectives[0]:
            self.loss_fn = BCELoss()
        else:
            self.loss_fn = BAYESIAN... # ToDo:

    def reset_parameters(self):
        self.embeddings.reset_parameters()

    def forward(self, edge_index: Adj, edge_label_index: Tensor, 
        x: Tensor = None, *args, **kwargs) -> Tensor:
        """"""
        x = self.embeddings(torch.arange(self.num_nodes))
        out = x

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            out = torch.add(out, x)

        out = torch.div(out, 4)
        nodes_first = torch.index_select(out, 0, edge_label_index[0,:].long())
        nodes_second = torch.index_select(out, 0, edge_label_index[1,:].long())
        out = torch.sum(nodes_first * nodes_second, dim=-1)

        if self.objective == LightGCN.objectives[1]:
            out = torch.sigmoid(out)
        
        return out

        log_prob = F.logsigmoid(x_uij).sum()
        regularization = self.weight_decay * (u.norm(dim=1).pow(2).sum() + i.norm(dim=1).pow(2).sum() + j.norm(dim=1).pow(2).sum())
        return -log_prob + regularization

    def recommend(self, u):
        """Return recommended item list given users.
        Args:
            u(torch.LongTensor): tensor stored user indexes. [batch_size,]
        Returns:
            pred(torch.LongTensor): recommended item list sorted by preference. [batch_size, item_size]
        """
        u = self.W[u, :]
        x_ui = torch.mm(u, self.H.t())
        pred = torch.argsort(x_ui, dim=1)
        return pred

    def loss(self, pred: Tensor, label: Tensor) -> float:
        return self.loss_fn(pred, label)

class BPRLoss(_Loss):
    r"""Bayesian Personalized Ranking (BPR) loss

    The Bayesian Personalized Ranking (BPR) loss is a pairwise loss that encourages
    the prediction of an observed entry to be higher than its unobserved counterparts
    (see the paper `"LightGCN: Simplifying and Powering Graph Convolution Network for
    Recommendation" <https://arxiv.org/abs/2002.02126>`_).

    .. math::
        L_{\text{BPR}} = - \sum_{u=1}^{M} \sum_{i \in \mathcal{N}_u} 
        \sum_{j \not\in \mathcal{N}_u} \ln \sigma(\hat{y}_{ui} - \hat{y}_{uj}) 
        + \lambda \vert\vert \textbf{x}^{(0)} \vert\vert^2

    where :math:`lambda` controls the :math:`L_2` regularization strength.
    
    Args:
        full (bool, optional): include the constant term in the loss
            calculation. Default: ``False``.
        eps (float, optional): value used to clamp ``var`` (see note below), for
            stability. Default: 1e-6.
        reduction (string, optional): specifies the reduction to apply to the
            output:``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
            will be applied, ``'mean'``: the output is the average of all batch
            member losses, ``'sum'``: the output is the sum of all batch member
            losses. Default: ``'mean'``.

    Shape:
        - Input: :math:`(N, *)` where :math:`*` means any number of additional
          dimensions
        - Target: :math:`(N, *)`, same shape as the input, or same shape as the input
          but with one dimension equal to 1 (to allow for broadcasting)
        - Var: :math:`(N, *)`, same shape as the input, or same shape as the input but
          with one dimension equal to 1, or same shape as the input but with one fewer
          dimension (to allow for broadcasting)
        - Output: scalar if :attr:`reduction` is ``'mean'`` (default) or
          ``'sum'``. If :attr:`reduction` is ``'none'``, then :math:`(N, *)`, same
          shape as the input

    Examples::
        >>> loss = nn.BPRLoss()
        >>> input = torch.randn(5, 2, requires_grad=True)
        >>> target = torch.randn(5, 2)
        >>> var = torch.ones(5, 2, requires_grad=True) #heteroscedastic
        >>> output = loss(input, target, var)
        >>> output.backward()

        >>> loss = nn.GaussianNLLLoss()
        >>> input = torch.randn(5, 2, requires_grad=True)
        >>> target = torch.randn(5, 2)
        >>> var = torch.ones(5, 1, requires_grad=True) #homoscedastic
        >>> output = loss(input, target, var)
        >>> output.backward()
    """
    __constants__ = ['lambda']
    lambda_reg: float

    def __init__(self, lambda_reg: int) -> None:
        super(BPRLoss, self).__init__(None, None, "sum")
        self.lambda_reg = lambda_reg

    def forward(self, input: Tensor, link_labels: Tensor) -> Tensor:
        return F.gaussian_nll_loss(input, target, var, full=self.full, eps=self.eps, reduction=self.reduction)