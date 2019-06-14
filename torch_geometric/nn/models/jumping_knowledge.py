import torch
from torch.nn import Linear, LSTM


class JumpingKnowledge(torch.nn.Module):
    r"""The Jumping Knowledge layer aggregation module from the
    `"Representation Learning on Graphs with Jumping Knowledge Networks"
    <https://arxiv.org/abs/1806.03536>`_ paper based on either
    **concatenation** (:obj:`"cat"`)

    .. math::

        \mathbf{x}_v^{(1)} \, \Vert \, \ldots \, \Vert \, \mathbf{x}_v^{(T)}

    **max pooling** (:obj:`"max"`)

    .. math::

        \max \left( \mathbf{x}_v^{(1)}, \ldots, \mathbf{x}_v^{(T)} \right)

    or **weighted summation**

    .. math::

        \sum_{t=1}^T \alpha_v^{(t)} \mathbf{x}_v^{(t)}

    with attention scores :math:`\alpha_v^{(t)}` obtained from a bi-directional
    LSTM (:obj:`"lstm"`).

    Args:
        mode (string): The aggregation scheme to use
            (:obj:`"cat"`, :obj:`"max"` or :obj:`"lstm"`).
        channels (int, optional): The number of channels per representation.
            Needs to be only set for LSTM-style aggregation.
            (default: :obj:`None`)
        num_layers (int, optional): The number of layers to aggregate. Needs to
            be only set for LSTM-style aggregation. (default: :obj:`None`)
    """

    def __init__(self, mode, channels=None, num_layers=None):
        super(JumpingKnowledge, self).__init__()
        self.mode = mode.lower()
        assert self.mode in ['cat', 'max', 'lstm']

        if mode == 'lstm':
            assert channels is not None, 'channels cannot be None for lstm'
            assert num_layers is not None, 'num_layers cannot be None for lstm'
            self.lstm = LSTM(
                channels, (num_layers * channels) // 2,
                bidirectional=True,
                batch_first=True)
            self.att = Linear(2 * ((num_layers * channels) // 2), 1)

        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'lstm'):
            self.lstm.reset_parameters()
        if hasattr(self, 'att'):
            self.att.reset_parameters()

    def forward(self, xs):
        r"""Aggregates representations across different layers.

        Args:
            xs (list or tuple): List containing layer-wise representations.
        """

        assert isinstance(xs, list) or isinstance(xs, tuple)

        if self.mode == 'cat':
            return torch.cat(xs, dim=-1)
        elif self.mode == 'max':
            return torch.stack(xs, dim=-1).max(dim=-1)[0]
        elif self.mode == 'lstm':
            x = torch.stack(xs, dim=1)  # [num_nodes, num_layers, num_channels]
            alpha, _ = self.lstm(x)
            alpha = self.att(alpha).squeeze(-1)  # [num_nodes, num_layers]
            alpha = torch.softmax(alpha, dim=-1)
            return (x * alpha.unsqueeze(-1)).sum(dim=1)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.mode)
