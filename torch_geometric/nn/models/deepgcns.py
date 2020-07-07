import torch
from torch.utils.checkpoint import checkpoint


class DeepGCNs(torch.nn.Module):
    r"""The skip connection operations from the
    `"DeepGCNs: Can GCNs Go as Deep as CNNs?"
    <https://arxiv.org/abs/1904.03751>`_ and `"All You Need to Train Deeper
    GCNs" <https://arxiv.org/abs/2006.07739>`_ papers.
    The implemented skip connections includes the pre-activation residual
    connection (:obj:`"res+"`), the residual connection (:obj:`"res+"`),
    the dense connection (:obj:`"dense"`) and no connections (:obj:`"plain"`).

    **Res+** (:obj:`"res+"`):

    .. math::
        \text{Normalization}\to\text{Activation}\to\text{Dropout}\to
        \text{GraphConv}\to\text{Res}

  or **Res/Dense/Plain** (:obj:`"res"`, :obj:`"dense"` or :obj:`"plain"`):

    .. math::
        \text{GraphConv}\to\text{Normalization}\to\text{Activation}\to
        \text{Res/Dense/Plain}\to\text{Dropout}

    Args:
        block (string, optional): The skip connection operation to use
            (:obj:`"res+"`, :obj:`"res"`, :obj:`"dense"` or :obj:`"plain"`).
            (default: :obj:`"res+"`)
    """
    def __init__(self, block='res+'):
        super(DeepGCNs, self).__init__()
        self.block = block.lower()
        assert self.block in ['res+', 'res', 'dense', 'plain']

    def forward(self, inputs, conv_layer=None, norm_layer=None, act_layer=None,
                dropout_layer=None, ckpt_grad=False):

        if self.block == 'res+':
            if norm_layer is not None:
                h = norm_layer(inputs[0])
            if act_layer is not None:
                h = act_layer(h)
            if dropout_layer is not None:
                h = dropout_layer(h)
            if conv_layer is not None:
                if ckpt_grad:
                    h = checkpoint(conv_layer, h, *inputs[1:])
                else:
                    h = conv_layer(h, *inputs[1:])

            out = h + inputs[0]

            return out

        elif self.block in ['res', 'dense', 'plain']:
            if conv_layer is not None:
                if ckpt_grad:
                    h = checkpoint(conv_layer, *inputs)
                else:
                    h = conv_layer(*inputs)
            if norm_layer is not None:
                h = norm_layer(h)
            if act_layer is not None:
                h = act_layer(h)

            if self.block == 'res':
                out = h + inputs[0]
            elif self.block == 'dense':
                out = torch.cat((h, inputs[0]), -1)
            elif self.block == 'plain':
                out = h
            else:
                raise Exception("Unknow block")

            if dropout_layer is not None:
                out = dropout_layer(out)

            return out

        else:
            raise Exception("Unknow block")

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.block)
