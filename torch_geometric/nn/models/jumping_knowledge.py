import torch
from torch.nn import Linear, LSTM


class JumpingKnowledge(torch.nn.Module):
    r""""""

    def __init__(self, mode, channels=None, num_layers=None):
        super(JumpingKnowledge, self).__init__()
        self.mode = mode.lower()
        assert self.mode in ['cat', 'max', 'lstm']

        if mode == 'lstm':
            assert channels is not None
            assert num_layers is not None
            self.lstm = LSTM(
                channels,
                channels * num_layers // 2,
                bidirectional=True,
                batch_first=True)
            self.att = Linear(2 * channels * num_layers // 2, 1)

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
