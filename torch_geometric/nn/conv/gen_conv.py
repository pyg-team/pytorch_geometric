import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Sequential as Seq, Linear as Lin
from .deepgcns_message import GenMessagePassing
from torch_geometric.nn.norm import MsgNorm
from torch_geometric.nn.conv.utils.helpers import act_layer, norm_layer
try:
    from ogb.graphproppred.mol_encoder import BondEncoder
except ImportError:
    print('BondEncoder not imported')


class MLP(Seq):
    def __init__(self, channels, act='relu',
                 norm=None, bias=True,
                 drop=0., last_lin=False):
        m = []

        for i in range(1, len(channels)):

            m.append(Lin(channels[i - 1], channels[i], bias))

            if (i == len(channels) - 1) and last_lin:
                pass
            else:
                if norm:
                    m.append(norm_layer(norm, channels[i]))
                if act:
                    m.append(act_layer(act))
                if drop > 0:
                    m.append(nn.Dropout2d(drop))

        self.m = m
        super(MLP, self).__init__(*self.m)


class GENConv(GenMessagePassing):
    r"""The GENeralized Graph Convolution (GENConv) from the `"DeeperGCN: All
    You Need to Train Deeper GCNs" <https://arxiv.org/abs/2006.07739>`_ paper.
    Support SoftMax  &  PowerMean Aggregation. The message construnction is:

    .. math::
        \mathbf{m}_{vu}^{(l)} = \bm{\rho}^{(l)}(\mathbf{h}_{v}^{(l)},
        \mathbf{h}_{u}^{(l)}, \mathbf{h}_{e_{vu}}^{(l)})=
        \text{ReLU}(\mathbf{h}_{u}^{(l)}+\mathbbm{1}(\mathbf{h}_{e_{vu}}^{(l)})
        \cdot\mathbf{h}_{e_{vu}}^{(l)})+\epsilon, ~u \in \mathcal{N}(v)

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
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
        msg_norm (bool, optinal): If set to :obj:`True`, will use message norm.
            (default: :obj:`False`)
        learn_msg_scale (bool, optional): If set to :obj:`True`, will learn the
            scaling factor of message norm. (default: :obj:`False`)
        encode_edge (bool, optional): If set to :obj:`True`, will encode the
            edge features. (default: :obj:`False`)
        bond_encoder (bool, optional): If set to :obj:`True`, will use the bond
            encoder to encode edge features. (default: :obj:`False`)
        edge_feat_dim (int, optional): Size of each edge features.
            (default: :obj:`None`)
        norm (str, optional): Norm layer of mlp layers. (default: :obj:`batch`)
        mlp_layers (int, optional): The num of mlp layers. (default: :obj:`2`)
        eps (float, optional): The epsilon value of the message construction
            function. (default: :obj:`1e-7`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.GenMessagePassing`.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 aggr: str = 'softmax',
                 t: float = 1.0, learn_t: bool = False,
                 p: float = 1.0, learn_p: bool = False,
                 msg_norm: bool = False, learn_msg_scale: bool = True,
                 encode_edge: bool = False, bond_encoder: bool = False,
                 edge_feat_dim=None,
                 norm: str = 'batch', mlp_layers: int = 2,
                 eps: float = 1e-7, **kwargs):

        super(GENConv, self).__init__(aggr=aggr,
                                      t=t, learn_t=learn_t,
                                      p=p, learn_p=learn_p, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        channels_list = [in_channels]

        for i in range(mlp_layers-1):
            channels_list.append(in_channels*2)

        channels_list.append(out_channels)

        self.mlp = MLP(channels=channels_list,
                       norm=norm,
                       last_lin=True)

        self.msg_encoder = torch.nn.ReLU()
        self.eps = eps

        self.msg_norm = msg_norm
        self.encode_edge = encode_edge
        self.bond_encoder = bond_encoder

        if msg_norm:
            self.msg_norm = MsgNorm(learn_msg_scale=learn_msg_scale)
        else:
            self.msg_norm = None

        if self.encode_edge:
            if self.bond_encoder:
                self.edge_encoder = BondEncoder(emb_dim=in_channels)
            else:
                self.edge_encoder = torch.nn.Linear(edge_feat_dim, in_channels)

    def forward(self, x, edge_index, edge_attr=None):
        x = x

        if self.encode_edge and edge_attr is not None:
            edge_emb = self.edge_encoder(edge_attr)
        else:
            edge_emb = edge_attr

        m = self.propagate(edge_index, x=x, edge_attr=edge_emb)

        if self.msg_norm is not None:
            m = self.msg_norm(x, m)

        h = x + m
        out = self.mlp(h)

        return out

    def message(self, x_j, edge_attr=None):

        if edge_attr is not None:
            msg = x_j + edge_attr
        else:
            msg = x_j

        return self.msg_encoder(msg) + self.eps

    def update(self, aggr_out):
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
