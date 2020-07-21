import torch
from torch import Tensor
from torch.nn import Parameter
import torch.nn.functional as F


class MessageNorm(torch.nn.Module):
    r"""Applies message normalization over the aggregated messages as described
    in the `"DeeperGCNs: All You Need to Train Deeper GCNs"
    <https://arxiv.org/abs/2006.07739>`_ paper

    .. math::
    \mathbf{h}_{v}^{(l+1)} = \bm{\phi}^{(l)}(\mathbf{h}_{v}^{(l)},
    \mathbf{m}_{v}^{(l)}) = \text{MLP}(\mathbf{h}_{v}^{(l)} + s \cdot
    \lVert\mathbf{h}_{v}^{(l)}\rVert_2 \cdot
    \frac{\mathbf{m}_{v}^{(l)}}{\lVert\mathbf{m}_{v}^{(l)}\rVert_2} )

    Args:
        learn_scale (bool, optional): If set to :obj:`True`, will learn the
            scaling factor of message normalization. (default: :obj:`False`)
    """
    def __init__(self, learn_scale: bool = False):
        super(MessageNorm, self).__init__()

        self.scale = Parameter(torch.Tensor([1.0]), requires_grad=learn_scale)

    def reset_parameters(self):
        self.scale.data.fill_(1.0)

    def forward(self, x: Tensor, msg: Tensor, p: int = 2):
        msg = F.normalize(msg, p=p, dim=-1)
        x_norm = x.norm(p=p, dim=-1, keepdim=True)
        return msg * x_norm * self.scale

    def __repr__(self):
        return ('{}(learn_scale={})').format(self.__class__.__name__,
                                             self.scale.requires_grad)
