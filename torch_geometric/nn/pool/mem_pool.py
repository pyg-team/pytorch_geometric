import torch
from torch import nn, matmul, Tensor, BoolTensor
from torch.nn.init import xavier_uniform_
from torch import nn


class MemPool(nn.Module):
    r"""Memory based pooling layer from `"MEMORY-BASED GRAPH NETWORKS"
    <https://arxiv.org/abs/2002.09518>`_ paper.

    .. math::
        \mathbf{S}{_i}{_j}=\frac{(1+\lVert {x_i-k_j} \rVert^2/\tau)^{
        -\frac{1+\tau}{2}}}{\sum_{j^{\prime}}(1
        +\lVert {x_i-k_{j^{\prime}}} \rVert^2/\tau)^{-\frac{1+\tau}{2}}}

        \mathbf{S}=\text{Softmax}(\Gamma_\phi(\lVert_{k=0}^H \mathbf{S}_i))

    Where :math:`\mathbf{H}` is the number of heads, with :math:`\mathbf{K}`
    trainable keys each. :math:`\Gamma_\phi` is a 1*1 convolution operation.
    Nodes are then pooled via.

    ..math::
        \mathbf{X}^\prime=\text{LeakyRelu}(\mathbf{S}^T\mathbf{XW})

    Args:
        in_channels (int): size of each input sample. :math:`F`.
        out_channels (int): size of each output sample. :math:`F\prime`.
        heads (int): number of heads. :math:`H`.
        num_keys (int): number of keys(clusters) per head. :math:`K`.
        tau (int): degrees of freedom.
    """
    def __init__(self, heads=1, num_keys=2, in_channels=2, out_channels=2,
                 tau=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.num_keys = num_keys
        self.tau = tau

        self.k = nn.Parameter(torch.rand(heads, num_keys, in_channels))
        self.conv = nn.Conv2d(heads, 1, kernel_size=1, padding=0, bias=False)
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.act = nn.LeakyReLU()
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()
        xavier_uniform_(self.lin.weight)

    @staticmethod
    def apply_mask(S: Tensor, mask: BoolTensor = None) -> Tensor:
        """"""
        S = S if mask is None else S * mask.unsqueeze(2)
        return S

    @staticmethod
    def computeKl(S: Tensor, mask: BoolTensor = None) -> (Tensor, Tensor):
        r"""Additional kl divergence based loss.
        ..math::
            \mathbf{P}{_i}{_j}=\frac{\mathbf{S}{_i}{_j}{^2}/\sum_i
            \mathbf{S}{_i}{_j}}{\sum{_j\prime}\mathbf{S}{_i}{_j\prime}^2/
            \sum_i\mathbf{S}{_i}{_j\prime}}

            \mathcal{L}_{kl}=\text{KLDiv}(\mathbf{P}||\mathbf{S})
        """
        S = MemPool.apply_mask(S, mask)  # B*N*K
        P = S**2 / S.sum(1).unsqueeze(1)
        denom = P.sum(2)
        if mask is not None:
            denom[~mask] = 1
        P = P / denom.unsqueeze(2)

        kl_loss = nn.KLDivLoss(reduction='sum')
        loss = kl_loss(S.log(), P)

        return loss, P

    def forward(self, x: Tensor, mask: BoolTensor = None) -> (Tensor, Tensor):
        r"""
        Args:
            x (Tensor): Node feature tensor :math:`\mathbf{X} \in \mathbb{R}^{B
                \times N \times F}` with batch-size :math:`B`, (maximum)
                number of nodes :math:`N` for each graph.
            mask (BoolTensor, optional): Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
                the valid nodes for each graph. (default: :obj:`None`)
        """
        x = x.unsqueeze(0) if x.dim() == 2 else x

        B = x.shape[0]
        H = self.k.shape[0]
        K = self.k.shape[1]
        N = x.shape[1]

        dist = torch.cdist(self.k.view(H * K, -1), x.view(B * N, -1), 1)
        dist = (1 + (dist**2 / self.tau)).pow(-(self.tau + 1) / 2.0)
        dist = dist.view(H, K, B, N).permute(2, 0, 3, 1)
        S = dist / dist.sum(3).unsqueeze(3)  # B*H*N*K

        S = self.conv(S).squeeze(1)
        S = torch.softmax(S, 2)  # B*N*K
        S = self.apply_mask(S, mask)
        x = self.lin(matmul(S.transpose(1, 2), x))  # B*K*F'
        x = self.act(x)

        return x, S

    def __repr__(self):
        return '{}({}, {}, heads={}, keys={})'.format(self.__class__.__name__,
                                                      self.in_channels,
                                                      self.out_channels,
                                                      self.heads,
                                                      self.num_keys)
