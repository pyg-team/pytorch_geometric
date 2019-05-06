import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter_add
from torch_geometric.utils import softmax
from torch_geometric.nn.conv import MessagePassing


class HypergraphConv(MessagePassing):
    def __init__(
        self,
        in_channels,
        out_channels,
        use_attention=False,
        head=1,
        negative_slope=0.2,
        dropout=0,
        aggregate_method="cat",
    ):
        r"""
        Code for Hypergraph Convolution in `"Hypergraph Convolution and
        Hypergraph Attention"<https://arxiv.org/pdf/1901.08150.pdf>`_paper   
        .. math::
            \mathbf{X}^{(l+1)}=\sigma\left(\mathbf{D}^{-1} \mathbf{H} \mathbf{W} \mathbf{B}^{-1}   
            \mathbf{H}^{\mathrm{T}} \mathbf{X}^{(l)} \mathbf{P}\right)  

        Args: 
            in_channels (int): channel of each input sample   
            out_channels (int): channel of each output sample   
            use_attention (bool,optional):If set to :obj:`True`,attention will be added to this layer (default: :obj:`False`)   
            head (int): head num of attention, this makes sense when use_attention is set to :obj:`True`   
            negative_slope (float): threshold of activation function in attention module   
            dropout (float): dropout ratio in attention module, default 0   
            aggregate_method (str): aggregation method to aggregate multiple head output for attention mode   
        """
        super().__init__("add", flow="target_to_source")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention

        self.linear = nn.Linear(in_channels, head * out_channels, bias=False)

        if self.use_attention:
            self.head = head
            self.alpha_initialized = False
            self.negative_slope = negative_slope
            self.dropout = dropout
            self.aggregate_method = aggregate_method
            self.attrs = nn.Parameter(torch.Tensor(1, head, 2 * out_channels))

    def norm(
        self, hyper_edge_index, dim_size, hyper_edge_weight=None, dim=0, dtype=None
    ):
        """
        Args:
            hyper_edge_index (Tensor): hype edge connect with node ,[2,E]   
            dim_size (int): size of degree   
            hyper_edge_weight (Tensor): weight of all hyper edges   
            dim (int): normalization in which dimension   
            dtype (torch.Dtype): data type of newly created tensor   
        :return: attention weight for each edge between nodes and hyper edges   
        """
        if hyper_edge_weight is None:
            hyper_edge_weight = torch.ones(
                (hyper_edge_index.size(1),), dtype=dtype, device=hyper_edge_index.device
            )
        else:
            hyper_edge_weight = torch.index_select(
                hyper_edge_weight, dim=0, index=hyper_edge_index[1]
            )
        if self.use_attention:
            hyper_edge_weight = hyper_edge_weight.view(-1, 1, 1) * self.attention_weight

        # hyper_edge_weight = hyper_edge_weight.view(-1)
        deg = scatter_add(
            hyper_edge_weight, hyper_edge_index[dim], dim=0, dim_size=dim_size
        )
        deg_inv = deg.pow(-1)
        deg_inv[deg_inv == float("inf")] = 0
        return torch.index_select(deg_inv, 0, hyper_edge_index[dim]) * hyper_edge_weight

    def attention(self, x, hyper_edge_index):
        """
        Args:
            x (Tensor): features for all nodes in a graph
            hyper_edge_index (Tensor): hype edge connect with node ,[2,E]
        """
        row, col = hyper_edge_index
        x_i = torch.index_select(x, dim=0, index=row)
        x_j = torch.index_select(x, dim=0, index=col)
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.attrs).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, row, x.size(0))
        if self.training and self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout, training=True)
        return alpha.view(-1, self.head, 1)

    def forward(self, x, hyper_edge_index, hyper_edge_weight):
        """
        Args:
            x (Tensor): node feature matrix, [N,C].
            hyper_edge_index (Tensor): hype edge connect with node ,[2,E]
            hyper_edge_weight (Tensor): weight for each unique hyper edge, [M,]
        """
        x = self.linear(x)
        if self.use_attention:
            # sim compute

            assert hyper_edge_weight.size(0) == x.size(
                0
            ), "Attention can only applied when hyper edge is node"
            x = x.view(-1, self.head, self.out_channels)
            self.attention_weight = self.attention(x, hyper_edge_index)
        norm_HW = self.norm(
            hyper_edge_index,
            x.size(0),
            hyper_edge_weight=hyper_edge_weight,
            dim=0,
            dtype=x.dtype,
        )
        norm_H = self.norm(
            hyper_edge_index, hyper_edge_weight.size(0), dim=1, dtype=x.dtype
        )
        #
        tmp = self.propagate(
            edge_index=hyper_edge_index,
            x=x,
            norm=norm_H,
            size=(hyper_edge_weight.size(0), x.size(0)),
        )  # M,C
        return self.propagate(
            edge_index=hyper_edge_index,
            x=tmp,
            norm=norm_HW,
            size=(x.size(0), hyper_edge_weight.size(0)),
        )  # N,C

    def message(self, x_j, norm):
        """
        Args:
            x_j (Tensor): features of connected nodes
            norm (Tensor): transition weights for connected nodes
        """
        if self.use_attention:
            return norm * x_j.view(-1, self.head, self.out_channels)
        return norm.view(-1, 1) * x_j  # E,1

    def att_aggregate(self, x):
        """
            x (Tensor): [N,head_num,C]
        """
        if self.aggregate_method == "cat":
            return x.view(-1, self.head * self.out_channels)
        elif self.aggregate_method == "avg":
            return x.mean(dim=1)
        else:
            raise NotImplementedError("Aggregation method %s is not implemented.")

    def update(self, aggr_out):
        """
        Args:
            aggr_out (Tensor): output of hyper graph layer
        """
        if self.use_attention:
            aggr_out = self.att_aggregate(aggr_out)
        return aggr_out

    def __repr__(self):
        return "{}({}, {})".format(
            self.__class__.__name__, self.in_channels, self.out_channels
        )
