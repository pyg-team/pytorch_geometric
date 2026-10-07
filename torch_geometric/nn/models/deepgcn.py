import inspect
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from torch.utils.checkpoint import checkpoint


class DeepGCNLayer(torch.nn.Module):
    r"""The skip connection operations from the
    `"DeepGCNs: Can GCNs Go as Deep as CNNs?"
    <https://arxiv.org/abs/1904.03751>`_ and `"All You Need to Train Deeper
    GCNs" <https://arxiv.org/abs/2006.07739>`_ papers.
    The implemented skip connections includes the pre-activation residual
    connection (:obj:`"res+"`), the residual connection (:obj:`"res"`),
    the dense connection (:obj:`"dense"`) and no connections (:obj:`"plain"`).

    * **Res+** (:obj:`"res+"`):

    .. math::
        \text{Normalization}\to\text{Activation}\to\text{Dropout}\to
        \text{GraphConv}\to\text{Res}

    * **Res** (:obj:`"res"`) / **Dense** (:obj:`"dense"`) / **Plain**
      (:obj:`"plain"`):

    .. math::
        \text{GraphConv}\to\text{Normalization}\to\text{Activation}\to
        \text{Res/Dense/Plain}\to\text{Dropout}

    .. note::

        For an example of using :obj:`GENConv`, see
        `examples/ogbn_proteins_deepgcn.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        ogbn_proteins_deepgcn.py>`_.

        To apply graph-wise normalization to a batch of disconnected graphs,
        pass the node-to-graph assignment vector via :obj:`norm_batch`::

            x = layer(x, edge_index, edge_type, norm_batch=batch)

        The vector is passed to normalization layers whose :meth:`forward`
        method accepts a :obj:`batch` argument. It is not passed to the graph
        convolution. If omitted, normalization is called as :obj:`norm(x)` as
        before. Other positional and keyword arguments are passed to the graph
        convolution.

    Args:
        conv (torch.nn.Module, optional): the GCN operator.
            (default: :obj:`None`)
        norm (torch.nn.Module): the normalization layer. (default: :obj:`None`)
        act (torch.nn.Module): the activation layer. (default: :obj:`None`)
        block (str, optional): The skip connection operation to use
            (:obj:`"res+"`, :obj:`"res"`, :obj:`"dense"` or :obj:`"plain"`).
            (default: :obj:`"res+"`)
        dropout (float, optional): Whether to apply or dropout.
            (default: :obj:`0.`)
        ckpt_grad (bool, optional): If set to :obj:`True`, will checkpoint this
            part of the model. Checkpointing works by trading compute for
            memory, since intermediate activations do not need to be kept in
            memory. Set this to :obj:`True` in case you encounter out-of-memory
            errors while going deep. (default: :obj:`False`)
    """
    def __init__(
        self,
        conv: Optional[Module] = None,
        norm: Optional[Module] = None,
        act: Optional[Module] = None,
        block: str = 'res+',
        dropout: float = 0.,
        ckpt_grad: bool = False,
    ):
        super().__init__()

        self.conv = conv
        self.norm = norm
        self.supports_norm_batch = False
        if norm is not None and hasattr(norm, 'forward'):
            norm_params = inspect.signature(norm.forward).parameters
            self.supports_norm_batch = 'batch' in norm_params
        self.act = act
        self.block = block.lower()
        assert self.block in ['res+', 'res', 'dense', 'plain']
        self.dropout = dropout
        self.ckpt_grad = ckpt_grad

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.conv.reset_parameters()
        self.norm.reset_parameters()

    def forward(
        self,
        *args,
        norm_batch: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        r"""Runs the graph convolution with the configured skip connection.

        Args:
            *args: The input node features, followed by additional positional
                arguments passed to the graph convolution.
            norm_batch (torch.Tensor, optional): The node-to-graph assignment
                vector used for normalization. Forwarded as :obj:`batch` when
                the normalization layer's inspectable :meth:`forward` signature
                explicitly declares a :obj:`batch` parameter. Otherwise,
                normalization is called without it. (default: :obj:`None`)
            **kwargs: Keyword arguments passed to :obj:`conv`.
        """
        args = list(args)
        x = args.pop(0)

        if self.block == 'res+':
            h = x
            if self.norm is not None:
                if norm_batch is not None and self.supports_norm_batch:
                    h = self.norm(h, batch=norm_batch)
                else:
                    h = self.norm(h)
            if self.act is not None:
                h = self.act(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            if self.conv is not None and self.ckpt_grad and h.requires_grad:
                h = checkpoint(self.conv, h, *args, use_reentrant=True,
                               **kwargs)
            else:
                h = self.conv(h, *args, **kwargs)

            return x + h

        else:
            if self.conv is not None and self.ckpt_grad and x.requires_grad:
                h = checkpoint(self.conv, x, *args, use_reentrant=True,
                               **kwargs)
            else:
                h = self.conv(x, *args, **kwargs)
            if self.norm is not None:
                if norm_batch is not None and self.supports_norm_batch:
                    h = self.norm(h, batch=norm_batch)
                else:
                    h = self.norm(h)
            if self.act is not None:
                h = self.act(h)

            if self.block == 'res':
                h = x + h
            elif self.block == 'dense':
                h = torch.cat([x, h], dim=-1)
            elif self.block == 'plain':
                pass

            return F.dropout(h, p=self.dropout, training=self.training)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(block={self.block})'
