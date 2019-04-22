import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax


class AGNNConv(MessagePassing):
    r"""Graph attentional propagation layer from the
    `"Attention-based Graph Neural Network for Semi-Supervised Learning"
    <https://arxiv.org/abs/1803.03735>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \mathbf{P} \mathbf{X},

    where the propagation matrix :math:`\mathbf{P}` is computed as

    .. math::
        P_{i,j} = \frac{\exp( \beta \cdot \cos(\mathbf{x}_i, \mathbf{x}_j))}
        {\sum_{k \in \mathcal{N}(i)\cup \{ i \}} \exp( \beta \cdot
        \cos(\mathbf{x}_i, \mathbf{x}_k))}

    with trainable parameter :math:`\beta`.

    Args:
        requires_grad (bool, optional): If set to :obj:`False`, :math:`\beta`
            will not be trainable. (default: :obj:`True`)
    """

    def __init__(self, requires_grad=True):
        super(AGNNConv, self).__init__('add')

        self.requires_grad = requires_grad

        if requires_grad:
            self.beta = Parameter(torch.Tensor(1))
        else:
            self.register_buffer('beta', torch.ones(1))

        self.reset_parameters()

    def reset_parameters(self):
        if self.requires_grad:
            self.beta.data.fill_(1)

    def forward(self, x, edge_index):
        """"""
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = add_self_loops(edge_index, x.size(0))

        x_norm = F.normalize(x, p=2, dim=-1)

        return self.propagate(
            edge_index, x=x, x_norm=x_norm, num_nodes=x.size(0))

    def message(self, edge_index_i, x_j, x_norm_i, x_norm_j, num_nodes):
        # Compute attention coefficients.
        beta = self.beta if self.requires_grad else self._buffers['beta']
        alpha = beta * (x_norm_i * x_norm_j).sum(dim=-1)
        alpha = softmax(alpha, edge_index_i, num_nodes)

        return x_j * alpha.view(-1, 1)

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
