from typing import Optional, Callable, Union, Tuple
from torch_geometric.typing import OptTensor, PairOptTensor, PairTensor, Adj

import torch
from torch import Tensor
from torch_sparse import SparseTensor, set_diag
from torch_scatter import scatter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops

class EquivariantConv(MessagePassing):
    r"""The Equivariant graph neural network operator form the
    `"E(n) Equivariant Graph Neural Networks"
    <https://arxiv.org/pdf/2102.09844.pdf>`_ paper
    ..math::
        \mathbf{m}_{ij}=h_{\mathbf{\Theta}}(\mathbf{x}_i,\mathbf{x}_j,\|
        {\mathbf{pos}_i-\mathbf{pos}_j}\|^2_2,\mathbf{e}_{ij})
        
        \mathbf{x}^{\prime}_i = \gamma_{\mathbf{\Theta}} \left( \mathbf{x}_i, 
        \sum_{j \in\mathcal{N}(i)} \mathbf{m}_{ij} \right)
        
        \mathbf{vel}^{\prime}_i = \phi_{\mathbf{\Theta}}(\mathbf{x}_i)\mathbf
        {vel}_i + \frac{1}{|\mathcal{N}(i)|}\sum_{j \in\mathcal{N}(i)} (\mathbf{pos}_i-\mathbf{pos}_j)
        \rho_{\mathbf{\Theta}}(\mathbf{m}_{ij})
        
        \mathbf{pos}^{\prime}_i = \mathbf{pos}_i + \mathbf{vel}_i
    
    where :math:`\gamma_{\mathbf{\Theta}}`, 
    :math:`h_{\mathbf{\Theta}}`, :math:\rho_{\mathbf{\Theta}}
    and :math:\phi_{\mathbf{\Theta}} denote neural
    networks, *.i.e.* MLPs. :math:`\mathbf{P} \in \mathbb{R}^{N \times D}`
    :math:`\mathbf{V} \in \mathbb{R}^{N \times D}`
    defines the position and velocity of each point respectively.
        

    Args:
        pos_nn (torch.nn.Module,optinal): A neural network that 
            maps message :obj:`` of shape :obj:`[-1, hidden_channels], 
            to shape :obj:`[-1, 1]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`. (default: :obj:`None`)
        vel_nn (torch.nn.Module,optional): A neural network that 
            maps node featues :obj:`x` of shape :obj:`[-1, in_channels], 
            to shape :obj:`[-1, 1]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`. (default: :obj:`None`)
        local_nn (torch.nn.Module, optional): A neural network
            :math:`h_{\mathbf{\Theta}}` that maps node features :obj:`x`,
            sqaured distance :obj:`pos_j - pos_i` and 
            edge_features :obj:`edge_attr`
            of shape :obj:`[-1, 2*in_channels + 1 +edge_dim]` 
            to shape :obj:`[-1, hidden_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`. (default: :obj:`None`)
        global_nn (torch.nn.Module, optional): A neural network
            :math:`\gamma_{\mathbf{\Theta}}` that maps aggregated node features
            of shape and node features :obj:`[-1, hidden_channels + in_channels]` 
            to shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`. (default: :obj:`None`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, pos_nn: Optional[Callable] = None,
                 vel_nn: Optional[Callable] = None,
                 local_nn: Optional[Callable] = None,
                 global_nn: Optional[Callable] = None,
                 add_self_loops: bool = True, aggr = "mean",**kwargs):
        super(EquivariantConv, self).__init__(**kwargs)

        self.pos_nn = pos_nn
        self.vel_nn = vel_nn
        self.local_nn = local_nn
        self.global_nn = global_nn
        self.add_self_loops = add_self_loops

        self.reset_parameters()

    def reset_parameters(self):
        if self.local_nn is not None:
            pass
        if self.global_nn is not None:
            pass

    def forward(self, x: Union[OptTensor,PairOptTensor],pos: Union[Tensor,PairTensor],
                edge_index: Adj, vel: OptTensor = None, 
                edge_attr: OptTensor = None) -> (Tensor,Tensor,OptTensor):
        """"""
        if not isinstance(x,Tuple):
            x:PairTensor = (x,x)
        
        if isinstance(pos,Tensor):
            pos:PairTensor = (pos,pos)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index,
                                               num_nodes=pos[1].size(0))
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        # propagate_type: (x: PairOptTensor, pos: PairTensor, edge_attr: OptTensor)
        out_x, out_pos = self.propagate(edge_index, x=x, pos=pos, size=None, 
                             edge_attr = edge_attr)
        
        out_x = out_x if x[1] is None else torch.cat([x[1],out_x],dim = 1)
        if self.global_nn is not None:
            out_x = self.global_nn(out_x)
            
        if vel  is None:
            out_pos += pos[1]
            out_vel = None
        else:
            if isinstance(vel,Tensor):
                vel:PairTensor =(vel,vel)
            out_vel = (vel[1] if self.vel_nn is None or x[1] is None
                       else vel[1]*self.vel_nn(x[1])) + out_pos
            out_pos = pos[1] + out_vel

        return out_x, out_pos, out_vel

    def message(self, x_i: Tensor, x_j: Tensor, pos_i: Tensor,
                pos_j: Tensor, edge_attr:OptTensor=None) -> Tensor:

        msg = torch.sum((pos_i - pos_j).square(),dim=1,keepdim=True)
        msg = msg if x_j is None else torch.cat([x_j, msg],dim=1)
        msg = msg if x_i is None else torch.cat([x_i, msg],dim=1)
        msg = msg if edge_attr is None else torch.cat([msg, 
                                                       edge_attr],dim=1)
        msg = msg if self.local_nn is None else self.local_nn(msg)
        
        pos_msg = ((pos_i-pos_j) if self.pos_nn is None 
                   else (pos_i-pos_j)*self.pos_nn(msg))
        return (msg,pos_msg)
    
    def aggregate(self,inputs: Tuple, index: Tensor):
        return (scatter(inputs[0],index, 0, reduce= self.aggr),
                scatter(inputs[1],index, 0, reduce= "mean"))
                
    def __repr__(self):
        return '{}(local_nn={}, global_nn={})'.format(self.__class__.__name__,
                                                      self.pos_nn,
                                                      self.vel_nn,
                                                      self.local_nn,
                                                      self.global_nn)

