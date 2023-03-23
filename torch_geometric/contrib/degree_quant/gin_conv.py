import torch
from torch_geometric.utils import remove_self_loops
from message_passing import *

class GINConvQuant(MessagePassingQuant):


    """
    A GIN Layer with complete quantization of all the parameters
    
    
    Args:
        nn (torch.nn.Module): A neural network defined by :class:`torch.nn.Sequential`.
        eps (float, optional): To change the importance of the message from original node.
         (default: :obj:`0.`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
        mp_quantizers (dict): A dictionary with the IntegerQuantizer defined for each Message Passing Layer parameter

        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    
    """

    def __init__(self, nn, eps=0, train_eps=False, mp_quantizers=None, **kwargs):
        super(GINConvQuant, self).__init__(
            aggr="add", mp_quantizers=mp_quantizers, **kwargs
        )
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer("eps", torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.nn.reset_parameters()
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x, edge_index, *args, **kwargs):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_index, _ = remove_self_loops(edge_index)
        out = self.propagate(edge_index, x=x)
        return out

    def message(self, x_j):
        return x_j

    def update(self, aggr_out, x):
        return self.nn((1 + self.eps) * x + aggr_out)

    def __repr__(self):
        return "{}(nn={})".format(self.__class__.__name__, self.nn)

    
class GINConvMultiQuant(MessagePassingMultiQuant):
    
    
    """
    A GCN Layer with Degree Quant approach for quantization of all the layer and message passing parameters
    It uses low and high masking strategy to quantize the respective quantizable tensors
    
    
    Args:
        nn (torch.nn.Module): A neural network defined by :class:`torch.nn.Sequential`.
        eps (float, optional): To change the importance of the message from original node.
         (default: :obj:`0.`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
        mp_quantizers (dict): A dictionary with the IntegerQuantizer defined for each Message Passing Layer parameter

        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    
    """

    def __init__(self, nn, eps=0, train_eps=False, mp_quantizers=None, **kwargs):
        super(GINConvMultiQuant, self).__init__(
            aggr="add", mp_quantizers=mp_quantizers, **kwargs
        )
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer("eps", torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.nn.reset_parameters()
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x, edge_index, mask):

        """
        Args:
            x (torch.Tensor): Node Features
            edge_index (torch.Tensor or SparseTensor): The tensor which is used to store the graph edges
            mask(torch.Tensor): The mask for the graph which is used to protect the nodes in the Degree Quant method
        
        """
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_index, _ = remove_self_loops(edge_index)
        out = self.propagate(edge_index, x=x, mask=mask)
        return out

    def message(self, x_j):
        return x_j

    def update(self, aggr_out, x):

        # We apply the post processing nn head here to the updated output of the layer 
        return self.nn((1 + self.eps) * x + aggr_out)

    def __repr__(self):
        return "{}(nn={})".format(self.__class__.__name__, self.nn)