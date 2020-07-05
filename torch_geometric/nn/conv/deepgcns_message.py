import torch
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter, scatter_softmax


class GenMessagePassing(MessagePassing):
    r"""The GENeralized message passing framework from the `"DeeperGCN: All
    You Need to Train Deeper GCNs" <https://arxiv.org/abs/2006.07739>`_ paper.
    Support SoftMax  &  PowerMean Aggregation.
    Args:
        aggr (str, optional): Message Aggregator supports :obj:`"softmax"`,
            :obj:`"softmax_sg"`, :obj:`"power"`, :obj:`"add"`, :obj:`"mean"`,
            :obj:`max`. (default: :obj:`"softmax"`)
        t (float, optional): Initial inverse temperature of softmax aggr.
            (default: :obj:`1.0`)
        learn_t (bool, optional): If set to :obj:`True`, will learn t for
            softmax aggr dynamically. (default: :obj:`False`)
        p (float, optional): Initial power for softmax aggr.
            (default: :obj:`1.0`)
        learn_p (bool, optional): If set to :obj:`True`, will learn p for
            power mean aggr dynamically. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, aggr='softmax',
                 t=1.0, learn_t=False,
                 p=1.0, learn_p=False, **kwargs):

        if aggr == 'softmax' or aggr == 'softmax_sg':

            super(GenMessagePassing, self).__init__(aggr=None, **kwargs)
            self.aggr = aggr

            if learn_t and aggr == 'softmax':
                if t < 1.0:
                    c = torch.nn.Parameter(torch.Tensor([1/t]),
                                           requires_grad=True)
                    self.t = 1 / c
                else:
                    self.t = torch.nn.Parameter(torch.Tensor([t]),
                                                requires_grad=True)
            else:
                self.t = t

        elif aggr == 'power':

            super(GenMessagePassing, self).__init__(aggr=None, **kwargs)
            self.aggr = aggr

            if learn_p:
                self.p = torch.nn.Parameter(torch.Tensor([p]),
                                            requires_grad=True)
            else:
                self.p = p
        else:
            super(GenMessagePassing, self).__init__(aggr=aggr, **kwargs)

    def aggregate(self, inputs, index, ptr=None, dim_size=None):

        if self.aggr in ['add', 'mean', 'max', None]:
            return super(GenMessagePassing, self).aggregate(inputs, index,
                                                            ptr, dim_size)

        elif self.aggr == 'softmax':
            out = scatter_softmax(inputs*self.t, index, dim=self.node_dim)
            out = scatter(inputs*out, index, dim=self.node_dim,
                          dim_size=dim_size, reduce='sum')
            return out

        elif self.aggr == 'softmax_sg':
            with torch.no_grad():
                out = scatter_softmax(inputs*self.t, index, dim=self.node_dim)
            out = scatter(inputs*out, index, dim=self.node_dim,
                          dim_size=dim_size, reduce='sum')
            return out

        elif self.aggr == 'power':
            min_value, max_value = 1e-7, 1e1
            torch.clamp_(inputs, min_value, max_value)
            out = scatter(torch.pow(inputs, self.p), index, dim=self.node_dim,
                          dim_size=dim_size, reduce='mean')
            torch.clamp_(out, min_value, max_value)
            return torch.pow(out, 1/self.p)

        else:
            raise NotImplementedError('To be implemented')

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.aggr)
