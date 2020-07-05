import torch
import torch.nn as nn
import torch.nn.functional as F


class MsgNorm(nn.Module):
    r"""Applies Message Normalization over the aggregated message as described
    in the `"DeeperGCNs: All You Need to Train Deeper GCNs"
    <https://arxiv.org/abs/2006.07739>`_ paper

    .. math::
    \mathbf{h}_{v}^{(l+1)} = \bm{\phi}^{(l)}(\mathbf{h}_{v}^{(l)},
    \mathbf{m}_{v}^{(l)}) = \text{MLP}(\mathbf{h}_{v}^{(l)} + s \cdot
    \lVert\mathbf{h}_{v}^{(l)}\rVert_2 \cdot
    \frac{\mathbf{m}_{v}^{(l)}}{\lVert\mathbf{m}_{v}^{(l)}\rVert_2} )

    Args:
        learn_msg_scale (bool, optional): If set to :obj:`True`, will learn the
            scaling factor of message norm. (default: :obj:`False`)
    """
    def __init__(self, learn_msg_scale=False):
        super(MsgNorm, self).__init__()

        self.msg_scale = nn.Parameter(torch.Tensor([1.0]),
                                      requires_grad=learn_msg_scale)

    def forward(self, x, msg, p=2):
        msg = F.normalize(msg, p=p, dim=1)
        x_norm = x.norm(p=p, dim=1, keepdim=True)
        msg = msg * x_norm * self.msg_scale
        return msg

    def __repr__(self):
        return ('{}({})').format(self.__class__.__name__,
                                 self.msg_scale.requires_grad)
