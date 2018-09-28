import torch


def dense_diff_pool(x, adj, s, mask=None):
    r"""Differentiable Pooling Operator based on dense learned assignments
    :math:`S` with

    .. math::
        \begin{align}
        F^{(l+1)} &= S^{(l)} X^{(l)}\\
        A^{(l+1)} &= {S^{(l)}}^{\top}A^{(l)} S^{(l)}
        \end{align}

    from the `"Hierarchical Graph Representation Learning with Differentiable
    Pooling" <https://arxiv.org/abs/1806.08804>`_ paper.
    """

    x = x.unsqueeze(0) if x.dim() == 2 else x
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s

    batch_size, num_nodes, _ = x.size()

    s = torch.softmax(s, dim=-1)

    if mask is not None:
        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x, s = x * mask, s * mask

    out = torch.matmul(s.transpose(1, 2), x)
    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

    reg = adj - torch.matmul(s, s.transpose(1, 2))
    reg = torch.norm(reg, p=2)
    reg = reg / adj.numel()

    return out, out_adj, reg
