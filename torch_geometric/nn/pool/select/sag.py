from typing import Callable, Optional, Union

import torch
from torch import Tensor

from torch_geometric.nn.pool.select import Select, SelectOutput
from torch_geometric.nn.pool.select.topk import topk
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.utils import softmax


class SelectSAG(Select):
    r"""Selects the top-:math:`k` nodes with highest projection scores from
    the `"Self-Attention Graph Pooling" <https://arxiv.org/abs/1904.08082>`_
    and `"Understanding Attention and Generalization in Graph Neural Networks"
    <https://arxiv.org/abs/1905.02850>`_ papers.

    If :obj:`min_score` :math:`\tilde{\alpha}` is :obj:`None`, computes:
        .. math::
            mathbf{i} &= \mathrm{top}_k(\mathbf{attn})

    If :obj:`min_score` :math:`\tilde{\alpha}` is a value in :obj:`[0, 1]`,
    computes:
        .. math::
            \mathbf{y} &= \mathrm{softmax}(\mathbf{attn})
            \mathbf{i} &= \mathbf{y}_i > \tilde{\alpha}

    Args:
        in_channels (int): Size of each input sample.
        ratio (float or int): The graph pooling ratio, which is used to compute
            :math:`k = \lceil \mathrm{ratio} \cdot N \rceil`, or the value
            of :math:`k` itself, depending on whether the type of :obj:`ratio`
            is :obj:`float` or :obj:`int`.
            This value is ignored if :obj:`min_score` is not :obj:`None`.
            (default: :obj:`0.5`)
        min_score (float, optional): Minimal node score :math:`\tilde{\alpha}`
            which is used to compute indices of pooled nodes
            :math:`\mathbf{i} = \mathbf{y}_i > \tilde{\alpha}`.
            When this value is not :obj:`None`, the :obj:`ratio` argument is
            ignored. (default: :obj:`None`)
        act (str or callable, optional): The non-linearity :math:`\sigma`.
            (default: :obj:`"tanh"`)
    """
    def __init__(self, ratio: Union[int, float] = 0.5,
                 min_score: Optional[float] = None,
                 act: Union[str, Callable] = 'tanh', *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if ratio is None and min_score is None:
            raise ValueError(f"At least one of the 'ratio' and 'min_score' "
                             f"parameters must be specified in "
                             f"'{self.__class__.__name__}'")

        self.ratio = ratio
        self.min_score = min_score
        self.act = activation_resolver(act)

    def forward(
        self,
        attn: Tensor,
        batch: Optional[Tensor] = None,
    ) -> SelectOutput:
        r"""Forward pass.

        Args:
            attn (torch.Tensor): The attention scores of modes.
            batch (torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each node to a specific example. (default: :obj:`None`)

        Return types:
            selectout (torch_geometric.nn.pool.select.SelectOutput): The object
                holds the output of the select part of SAGPooling.
        """
        if batch is None:
            batch = attn.new_zeros(attn.size(0), dtype=torch.long)

        attn = attn.view(-1, 1) if attn.dim() == 1 else attn
        attn = attn.squeeze(-1)
        if self.min_score is None:
            score = self.act(attn)
        else:
            score = softmax(attn, batch)

        node_index = topk(score, self.ratio, batch, self.min_score)

        return SelectOutput(
            node_index=node_index,
            num_nodes=score.size(0),
            cluster_index=torch.arange(node_index.size(0),
                                       device=score.device),
            num_clusters=node_index.size(0),
            weight=score[node_index],
        )
