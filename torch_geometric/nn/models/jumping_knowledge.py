from typing import Dict, List, Optional

import torch
from torch import Tensor
from torch.nn import LSTM, Linear


class JumpingKnowledge(torch.nn.Module):
    r"""The Jumping Knowledge layer aggregation module from the
    `"Representation Learning on Graphs with Jumping Knowledge Networks"
    <https://arxiv.org/abs/1806.03536>`_ paper.

    Jumping knowledge is performed based on either **concatenation**
    (:obj:`"cat"`)

    .. math::

        \mathbf{x}_v^{(1)} \, \Vert \, \ldots \, \Vert \, \mathbf{x}_v^{(T)},

    **max pooling** (:obj:`"max"`)

    .. math::

        \max \left( \mathbf{x}_v^{(1)}, \ldots, \mathbf{x}_v^{(T)} \right),

    or **weighted summation**

    .. math::

        \sum_{t=1}^T \alpha_v^{(t)} \mathbf{x}_v^{(t)}

    with attention scores :math:`\alpha_v^{(t)}` obtained from a bi-directional
    LSTM (:obj:`"lstm"`).

    Args:
        mode (str): The aggregation scheme to use
            (:obj:`"cat"`, :obj:`"max"` or :obj:`"lstm"`).
        channels (int, optional): The number of channels per representation.
            Needs to be only set for LSTM-style aggregation.
            (default: :obj:`None`)
        num_layers (int, optional): The number of layers to aggregate. Needs to
            be only set for LSTM-style aggregation. (default: :obj:`None`)
    """
    def __init__(self, mode: str, channels: Optional[int] = None,
                 num_layers: Optional[int] = None):
        super().__init__()
        self.mode = mode.lower()
        assert self.mode in ['cat', 'max', 'lstm']

        if mode == 'lstm':
            assert channels is not None, 'channels cannot be None for lstm'
            assert num_layers is not None, 'num_layers cannot be None for lstm'
            self.lstm = LSTM(channels, (num_layers * channels) // 2,
                             bidirectional=True, batch_first=True)
            self.att = Linear(2 * ((num_layers * channels) // 2), 1)
            self.channels = channels
            self.num_layers = num_layers
        else:
            self.lstm = None
            self.att = None
            self.channels = None
            self.num_layers = None

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        if self.lstm is not None:
            self.lstm.reset_parameters()
        if self.att is not None:
            self.att.reset_parameters()

    def forward(self, xs: List[Tensor]) -> Tensor:
        r"""Forward pass.

        Args:
            xs (List[torch.Tensor]): List containing the layer-wise
                representations.
        """
        if self.mode == 'cat':
            return torch.cat(xs, dim=-1)
        elif self.mode == 'max':
            return torch.stack(xs, dim=-1).max(dim=-1)[0]
        else:  # self.mode == 'lstm'
            assert self.lstm is not None and self.att is not None
            x = torch.stack(xs, dim=1)  # [num_nodes, num_layers, num_channels]
            alpha, _ = self.lstm(x)
            alpha = self.att(alpha).squeeze(-1)  # [num_nodes, num_layers]
            alpha = torch.softmax(alpha, dim=-1)
            return (x * alpha.unsqueeze(-1)).sum(dim=1)

    def __repr__(self) -> str:
        if self.mode == 'lstm':
            return (f'{self.__class__.__name__}({self.mode}, '
                    f'channels={self.channels}, layers={self.num_layers})')
        return f'{self.__class__.__name__}({self.mode})'


class HeteroJK(torch.nn.Module):
    r"""Heterogeneous Jumping Knowledge, this class expands the usage of
    :class:`~torch_geometric.nn.models.JumpingKnowledge` to heterogeneous
    graph features.


    Args:
        node_types (List[str]): List of node types. This can be dynamically
            inferred using :obj:`torch_geometric.data.HeteroData.node_types`.
        mode (str): The aggregation scheme to use
            (:obj:`"cat"`, :obj:`"max"` or :obj:`"lstm"`).
        kwargs (optional): Additional arguments to be forwarded in the case of
            the LSTM style aggregation. Namely you will need to provide the
            following kwargs :obj:`channels` and :obj:`num_layers`. Please
            refer to the documentation on
            :class:`~torch_geometric.nn.models.JumpingKnowledge` for further
            details.
    """
    def __init__(self, node_types: List[str], mode: str, **kwargs):
        super().__init__()
        self.jk_dict = torch.nn.ModuleDict()
        self.mode = mode.lower()
        for node_type in node_types:
            self.jk_dict[node_type] = JumpingKnowledge(mode=self.mode,
                                                       **kwargs)

    def forward(self, xs_dict: Dict[str, List[Tensor]]) -> Dict[str, Tensor]:
        r"""Forward pass.

        Args:
            xs_dict (Dict[str, List[torch.Tensor]]): A dictionary holding a
                list of layer-wise representation for each node type.
        """
        out_dict = {}
        for node_type, jk in self.jk_dict.items():
            out_dict[node_type] = jk(xs_dict[node_type])
        return out_dict

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for node_t in self.jk_dict:
            self.jk_dict[node_t].reset_parameters()

    def __repr__(self):
        if self.mode == 'lstm':
            ref_jk = next(iter(self.jk_dict.values()))
            return (f'{self.__class__.__name__}('
                    f'num_node_types={len(self.jk_dict)}'
                    f', mode={self.mode}, channels={ref_jk.channels}'
                    f', layers={ref_jk.num_layers})')
        return (f'{self.__class__.__name__}(num_node_types={len(self.jk_dict)}'
                f', mode={self.mode})')
