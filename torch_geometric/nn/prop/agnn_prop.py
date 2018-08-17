import torch
from torch.nn import Parameter
from torch_sparse import spmm
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax


class AGNNProp(torch.nn.Module):
    """Graph Attentional Propagation Layer from the
    `"Attention-based Graph Neural Network for Semi-Supervised Learning (AGNN)"
    <https://arxiv.org/abs/1803.03735>`_ paper.

    Args:
        requires_grad (bool, optional): If set to :obj:`False`, the propagation
            layer will not be trainable. (default: :obj:`True`)
    """

    def __init__(self, requires_grad=True):
        super(AGNNProp, self).__init__()

        if requires_grad:
            self.beta = Parameter(torch.Tensor(1))
        else:
            self.register_buffer('beta', torch.ones(1))

        self.requires_grad = requires_grad
        self.reset_parameters()

    def reset_parameters(self):
        if self.requires_grad:
            self.beta.data.uniform_(0, 1)

    def forward(self, x, edge_index):
        num_nodes = x.size(0)

        x = x.unsqueeze(-1) if x.dim() == 1 else x
        beta = self.beta if self.requires_grad else self._buffers['beta']

        # Add self-loops to adjacency matrix.
        edge_index, edge_attr = remove_self_loops(edge_index)
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))
        row, col = edge_index

        # Compute attention coefficients.
        norm = torch.norm(x, p=2, dim=1)
        alpha = (x[row] * x[col]).sum(dim=1) / (norm[row] * norm[col])
        alpha = softmax(alpha * beta, row, num_nodes=x.size(0))

        # Perform the propagation.
        out = spmm(edge_index, alpha, num_nodes, x)

        return out

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
