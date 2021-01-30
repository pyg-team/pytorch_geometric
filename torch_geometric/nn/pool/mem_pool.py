import torch
from torch import nn
from torch.nn.init import kaiming_uniform_, xavier_uniform_


class MemPool(nn.Module):
    r"""Memory based pooling from `"MEMORY-BASED GRAPH NETWORKS"
    <https://arxiv.org/abs/2002.09518>`_ paper.
    Does not use Adjacency matrix while pooling.

    .. math::
        C{_i}{_j}=\frac{(1+\lVert {q_i-k_j} \rVert^2/\tau)^{
        -\frac{1+\tau}{2}}}{\sum_{j^{\prime}}(1
        +\lVert {q_i-k_{j^{\prime}}} \rVert^2/\tau)^{-\frac{1+\tau}{2}}}

    if there are h heads. A tensor of cluster assignments is created.
    .. math::
        [C^0,C^1..C^{heads}]

    A 1*1 conv followed by a softmax
    applied to get a single assginment matrix.
    .. math::
        C^l=softmax(\Gamma_\phi(\lVert_{k=0}^h C_k))\quad \Gamma_\phi
        \enspace is \enspace convolution\enspace operator

    Pooled feature matrx.
    .. math::
        V^l=C^TQ^{l}
        Q^{l+1}=leakyRelu(V^lW)
    Args:
    heads (int): number of heads in keys.
    keys (int): number of clusters.
    in_feat (int): input feature dimension.
    out (int): output feature dimension.
    tau (int): degrees of freedom.
    """
    def __init__(self, heads=1, keys=2, in_feat=2, out_feat=2, tau=1):
        super().__init__()
        self.heads = heads
        self.out_feat = out_feat
        self.tau = tau

        self.k = nn.Parameter(torch.rand(heads, keys, in_feat))
        self.conv = nn.Parameter(torch.rand(heads, 1))
        self.w = nn.Parameter(torch.rand(in_feat, out_feat))
        self.activation = nn.LeakyReLU()
        self.initialize()

    def initialize(self):

        xavier_uniform_(self.conv)
        kaiming_uniform_(self.w)

    def maskC(self, C, mask=None):

        if mask is not None:
            m = torch.unsqueeze(mask, 1).repeat(1, C.shape[1], 1)
            C = C * m

        return C

    def computeKl(self, C, mask=None):
        r"""Additional kl divergence based loss
        Args:
        C (tensor): output assignment matrix from forward().
        mask (tensor): :math:`{0,1}^{B,N}`.
        :rtype: (:class:`Tensor`, :class:`Tensor`)
        """
        C = self.maskC(C, mask)  # batch*cluster*nodes

        P = torch.pow(C, 2) / torch.unsqueeze(torch.sum(C, 2), 2)
        denom = torch.sum(P, 1)
        if mask is not None:
            denom[~mask.bool()] = 1
        P = P / torch.unsqueeze(denom, 1)

        if mask is not None:
            m = torch.unsqueeze(mask, 1).repeat(1, C.shape[1], 1)
            # assigned to 1. masked nodes will have 0 loss.
            P[~m.bool()] = 1
            C[~m.bool()] = 1

        loss = P * torch.log(P / C)
        loss = torch.sum(loss)

        if mask is not None:
            P[~m.bool()] = 0
            C[~m.bool()] = 0

        return loss, P

    def forward(self, q, mask=None):
        r"""
        :rtype: (:class:`Tensor`, :class:`Tensor`)
        """
        if q.dim() == 2:
            q = q.unsqueeze(0)
        if mask is not None and mask.dim() == 1:
            mask = mask.unsqueeze(0)
        batch = q.shape[0]
        heads = self.k.shape[0]
        clusters = self.k.shape[1]
        nodes = q.shape[1]

        # shape: batch*heads*clusters*nodes*features
        k_broad = torch.unsqueeze(self.k,
                                  2).repeat(1, 1, nodes,
                                            1).repeat(batch, 1, 1, 1, 1)
        q_broad = torch.unsqueeze(torch.unsqueeze(q, 1),
                                  2).repeat(1, heads, clusters, 1, 1)

        dist = torch.pow(
            1 + torch.sum(torch.pow(q_broad - k_broad, 2), 4) / self.tau,
            -(self.tau + 1) / 2)
        denom = torch.unsqueeze(torch.sum(dist, 2),
                                2).repeat(1, 1, clusters, 1)

        C = dist / denom
        C = torch.matmul(C.permute(0, 3, 2, 1), self.conv).permute(0, 3, 2, 1)
        C = torch.squeeze(C)
        C = torch.softmax(C, 1)
        C = self.maskC(C, mask)  # batch*clusters*nodes

        q_l = torch.matmul(C, q)
        v = self.activation(torch.matmul(q_l, self.w))

        return v, C
