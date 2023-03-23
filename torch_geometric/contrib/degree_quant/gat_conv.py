
import torch 
from torch.nn import Parameter, ModuleDict
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import  add_self_loops, remove_self_loops, softmax
from message_passing import *


class GATConvQuant(MessagePassingQuant):
  
    """
    A GAT Layer with complete quantization of all the parameters

    Args:
    in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
        derive the size from the first input(s) to the forward method.
        A tuple corresponds to the sizes of source and target
        dimensionalities.
    out_channels (int): Size of each output sample.
    nn(Sequential, optional): A sequential layer to be used on the layer outputs

    heads (int, optional): Number of multi-head-attentions.
        (default: :obj:`1`)
    concat (bool, optional): If set to :obj:`False`, the multi-head
        attentions are averaged instead of concatenated.
        (default: :obj:`True`)
    negative_slope (float, optional): LeakyReLU angle of the negative
        slope. (default: :obj:`0.2`)
    dropout (float, optional): Dropout probability of the normalized
        attention coefficients which exposes each node to a stochastically
        sampled neighborhood during training. (default: :obj:`0`)
    bias (bool, optional): If set to :obj:`False`, the layer will not learn
        an additive bias. (default: :obj:`True`)
    
    mp_quantizers (dict): A dictionary with the IntegerQuantizer defined for each Message Passing Layer weight
    layer_quantizers (dict): A dictionary with the IntegerQuantizer defined for each layer trainable parameter

    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.MessagePassing`.
    
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        nn=None,
        heads=1,
        concat=True,
        negative_slope=0.2,
        dropout=0,
        bias=True,
        mp_quantizers=None,
        layer_quantizers=None,
        **kwargs,
    ):
        super(GATConvQuant, self).__init__(aggr="add", mp_quantizers=mp_quantizers, **kwargs)
        self.nn = nn
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.weight = Parameter(torch.Tensor(in_channels, heads * out_channels))
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        assert layer_quantizers is not None
        self.layer_quant_fns = layer_quantizers
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        glorot(self.weight)
        glorot(self.att)
        zeros(self.bias)
        self.nn.reset_parameters()

        self.layer_quantizers = ModuleDict()
        for key in ["weights","inputs","features","attention","alpha"]:
            self.layer_quantizers[key] = self.layer_quant_fns[key]()

    def forward(self, x, edge_index, size=None):

        # quantizing input
        x_q = self.layer_quantizers["inputs"](x)

        # quantizing layer weights
        w_q = self.layer_quantizers["weights"](self.weight)

        if size is None and torch.is_tensor(x_q):
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=x_q.size(0))

        if torch.is_tensor(x_q):
            x = torch.matmul(x_q, w_q)
            x_q = self.layer_quantizers["features"](x)
        else:
            x = (
                None if x_q[0] is None else torch.matmul(x_q[0], w_q),
                None if x_q[1] is None else torch.matmul(x_q[1], w_q),
            )

            x_q = (
                None if x[0] is None else self.layer_quantizers["features"](x[0]),
                None if x[1] is None else self.layer_quantizers["features"](x[1]),
            )

        return self.propagate(edge_index, size=size, x=x_q)

    def message(self, edge_index_i, x_i, x_j, size_i):
        # Compute attention coefficients.
        x_j = x_j.view(-1, self.heads, self.out_channels)
        att = self.layer_quantizers["attention"](self.att)

        if x_i is None:
            alpha = (x_j * att[:, :, self.out_channels :]).sum(dim=-1)
        else:
            x_i = x_i.view(-1, self.heads, self.out_channels)
            alpha = (torch.cat([x_i, x_j], dim=-1) * att).sum(dim=-1)

        alpha = self.layer_quantizers["alpha"](alpha)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        
        alpha = softmax(alpha, edge_index_i, size_i)
        
       
        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return self.nn(aggr_out)

    def __repr__(self):
        return "{}({}, {}, heads={})".format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.heads
        )

    
    
class GATConvMultiQuant(MessagePassingMultiQuant):
    
    
    """
    A GAT Layer with Degree Quant approach for quantization of all the layer and message passing parameters
    It uses low and high masking strategy to quantize the respective quantizable tensors

    Args:
    in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
        derive the size from the first input(s) to the forward method.
        A tuple corresponds to the sizes of source and target
        dimensionalities.
    out_channels (int): Size of each output sample.
    nn(Sequential, optional): A sequential layer to be used on the layer outputs

    heads (int, optional): Number of multi-head-attentions.
        (default: :obj:`1`)
    concat (bool, optional): If set to :obj:`False`, the multi-head
        attentions are averaged instead of concatenated.
        (default: :obj:`True`)
    negative_slope (float, optional): LeakyReLU angle of the negative
        slope. (default: :obj:`0.2`)
    dropout (float, optional): Dropout probability of the normalized
        attention coefficients which exposes each node to a stochastically
        sampled neighborhood during training. (default: :obj:`0`)
    bias (bool, optional): If set to :obj:`False`, the layer will not learn
        an additive bias. (default: :obj:`True`)
    
    mp_quantizers (dict): A dictionary with the IntegerQuantizer defined for each Message Passing Layer weight
    layer_quantizers (dict): A dictionary with the IntegerQuantizer defined for each layer trainable parameter

    **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.MessagePassing`.
    
    """
    
    def __init__(
        self,
        in_channels,
        out_channels,
        nn = None,
        heads=1,
        concat=True,
        negative_slope=0.2,
        dropout=0,
        bias=True,
        layer_quantizers=None,
        mp_quantizers=None,
        **kwargs,
    ):
        
       
        super(GATConvMultiQuant, self).__init__(
            aggr="add", mp_quantizers=mp_quantizers, **kwargs
        )
        self.nn = nn
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.weight = Parameter(torch.Tensor(in_channels, heads * out_channels))
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        assert layer_quantizers is not None
        self.layer_quant_fns = layer_quantizers
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        glorot(self.weight)
        glorot(self.att)
        zeros(self.bias)
        self.nn.reset_parameters()
        # create quantization modules for this layer
        self.layer_quantizers = ModuleDict()
        for key in ["weights_low","inputs_low","inputs_high","features_low","features_high","attention_low","alpha_low","alpha_high"]:
            self.layer_quantizers[key] = self.layer_quant_fns[key]()

    def forward(self, x, edge_index, mask, size=None):
        
        """
        Args:
            x (torch.Tensor): Node Features
            edge_index (torch.Tensor or SparseTensor): The tensor which is used to store the graph edges
            mask(torch.Tensor): The mask for the graph which is used to protect the nodes in the Degree Quant method
        
        """
        # quantizing input
        if self.training:
            x_q = torch.empty_like(x)
            x_q[mask] = self.layer_quantizers["inputs_high"](x[mask])
            x_q[~mask] = self.layer_quantizers["inputs_low"](x[~mask])
        else:
            x_q = self.layer_quantizers["inputs_low"](x)

        # quantizing layer weights
        w_q = self.layer_quantizers["weights_low"](self.weight)

        if size is None and torch.is_tensor(x_q):
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=x_q.size(0))

        if torch.is_tensor(x_q):
            if self.training:
                x = torch.empty((x_q.shape[0], w_q.shape[1])).to(x_q.device)
                x_tmp = torch.matmul(x_q, w_q)
                x[mask] = self.layer_quantizers["features_high"](x_tmp[mask])
                x[~mask] = self.layer_quantizers["features_low"](x_tmp[~mask])
            else:
                x = self.layer_quantizers["features_low"](torch.matmul(x_q, w_q))

            x_q = x
        else:
            x = (
                None if x_q[0] is None else torch.matmul(x_q[0], w_q),
                None if x_q[1] is None else torch.matmul(x_q[1], w_q),
            )
            if self.training:
                x0_q = None
                if x[0] is not None:
                    x0_q = torch.empty_like(x[0])
                    x0_q[mask] = self.layer_quantizers["features_high"](x[0][mask])
                    x0_q[~mask] = self.layer_quantizers["features_low"](x[0][~mask])

                x1_q = None
                if x[1] is not None:
                    x1_q = torch.empty_like(x[1])
                    x1_q[mask] = self.layer_quantizers["features_high"](x[1][mask])
                    x1_q[~mask] = self.layer_quantizers["features_low"](x[1][~mask])

                x_q = (x0_q, x1_q)

            else:
                x_q = (
                    None
                    if x[0] is None
                    else self.layer_quantizers["features_low"](x[0]),
                    None
                    if x[1] is None
                    else self.layer_quantizers["features_low"](x[1]),
                )

        edge_mask = torch.index_select(mask, 0, edge_index[0])
        return self.propagate(
            edge_index, size=size, x=x_q, mask=mask, edge_mask=edge_mask
        )

    def message(self, edge_index_i, x_i, x_j, size_i, edge_mask):
        x_j = x_j.view(-1, self.heads, self.out_channels)
        att = self.layer_quantizers["attention_low"](self.att)

        if x_i is None:
            alpha = (x_j * att[:, :, self.out_channels :]).sum(dim=-1)
        else:
            x_i = x_i.view(-1, self.heads, self.out_channels)
            alpha = (torch.cat([x_i, x_j], dim=-1) * att).sum(dim=-1)

        if self.training:
            alpha_tmp = torch.empty_like(alpha)
            alpha_tmp[edge_mask] = self.layer_quantizers["alpha_high"](alpha[edge_mask])
            alpha_tmp[~edge_mask] = self.layer_quantizers["alpha_low"](
                alpha[~edge_mask]
            )
            alpha = alpha_tmp
        else:
            alpha = self.layer_quantizers["alpha_low"](alpha)

        alpha = F.leaky_relu(alpha, self.negative_slope)

        if torch.cuda.is_available():
            alpha = softmax(alpha.cpu(), edge_index_i.cpu()).to(torch.device('cuda'))
        else: 
            alpha = softmax(alpha, edge_index_i)


        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        
        # We apply the post processing nn head here to the updated output of the layer 
        return self.nn(aggr_out)

    def __repr__(self):
        return "{}({}, {}, heads={})".format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.heads
        )
    
    