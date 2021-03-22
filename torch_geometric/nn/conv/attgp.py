import torch
from torch import Tensor

from torch.nn import Linear, BatchNorm1d, Dropout
from torch.nn import Parameter as Param
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, EdgePooling
from torch_sparse import matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from torch_geometric.typing import PairTensor, Adj, OptTensor, Size
from torch_scatter import scatter_add, scatter_max
import pickle

from typing import Union, Tuple, Optional

class GatConvAtom(MessagePassing):
    """
    This function does only the atom embedding, not the molecule embedding
    """
    def __init__(self, atom_in_channels: int, bond_in_channels: int, fingerprint_dim: int, dropout: float, bias: bool = True, debug: bool = False, step = 0, **kwargs):
        super(GatConvAtom, self).__init__()

        self.atom_in_channels = atom_in_channels
        self.bond_in_channels = bond_in_channels
        self.fingerprint_dim = fingerprint_dim
        self.step = step

        if  self.step == 0 : 
            self.atom_fc = Linear(atom_in_channels, fingerprint_dim, bias=bias)
            self.neighbor_fc = Linear(atom_in_channels + bond_in_channels, fingerprint_dim, bias=bias)
        self.align = Linear(2*fingerprint_dim, 1, bias=bias)
        self.attend = Linear(fingerprint_dim, fingerprint_dim, bias=bias)
        self.debug = debug
        self.dropout = Dropout(p=dropout)
        self.rnn = torch.nn.GRUCell(fingerprint_dim, fingerprint_dim)

        
    def forward(self, x: Union[Tensor,PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        
        out = self.propagate(edge_index, x = x, edge_attr=edge_attr, size=size)
        return out

    def message(self, x_i, x_j, edge_index: Adj, edge_attr: OptTensor, size) -> Tensor:

        if self.debug:
            print('a x_j:',x_j.shape,'x_i:',x_i.shape,'edge_attr:',edge_attr.shape)
        if  self.step == 0 :

            x_i = F.leaky_relu(self.atom_fc(x_i)) # code 3 

            # neighbor_feature => neighbor_fc
            x_j = torch.cat([x_j, edge_attr], dim=-1) # code 8
            if self.debug:
                print('b neighbor_feature i = 0', x_j.shape)
            
            x_j = F.leaky_relu(self.neighbor_fc(x_j)) # code 9
            if self.debug:
                print('c neighbor_feature i = 0', x_j.shape)
            
        # align score
        evu = F.leaky_relu(self.align(torch.cat([x_i, x_j], dim=-1))) # code 10
        if self.debug:
            print('d align_score:', evu.shape)
        
        avu = softmax(evu, edge_index[0], None, x_i.size(0))
        
        if self.debug:
            print('e attention_weight:', avu.shape)

        c_i = F.elu(torch.mul(avu, self.attend(self.dropout(x_i)))) # code 12

        if self.debug:
            print('f context',c_i.shape)
            
        x_i = self.rnn(c_i, x_i)
        if self.debug:
            print('g gru',c_i.shape)            

        return x_i   


class GatSuperNode(MessagePassing):
    """
    This function does the supernode embedding there is no convolution for the supernode (it takes all the atoms!)
    """ 
    def __init__(self, fingerprint_dim: int, dropout: int, debug: bool = False, step = 0):
        super(GatSuperNode, self).__init__()
        # need to find the correct dimensions 
        self.step = step
        self.mol_align = Linear(2*fingerprint_dim,1)
        self.mol_attend = Linear(fingerprint_dim,fingerprint_dim)
        self.dropout = Dropout(p=dropout)
        self.debug = debug
        self.rnn = torch.nn.GRUCell(fingerprint_dim, fingerprint_dim)

    def forward(self, h_s, x, batch):
        if self.debug:
            print('0 h_s / x dims:',h_s.shape, x.shape)
        # expanding at atom dim the supernode aggregation (ie SUM over atoms of a molecule not over the fingerprint dim!!!)
        h_s_ex= h_s[batch]

        esv = F.leaky_relu(self.mol_align(torch.cat([h_s_ex, x], dim=-1))) # code 5
        if self.debug:
            print('1 mol_align_score:',esv.shape)
        # this is a sotfmax per molecule  
        # error it's wrong again
        #asv = F.softmax(esv, dim=-1) # code 6
        #asv = self.softmax(esv, batch, num=batch.max()+1)
        
        superatom_num =  batch.max()+1
        asv = softmax(esv, batch, None,superatom_num)

        if self.debug:
            print('2 mol_align_score:',asv.shape)
        
        # this is not correct it should be more hs and not x_i there based on the paper supplementary table 3!
        # in the paper it's h_s_ex in the pytorch it's x !
        cs_i = scatter_add( torch.mul(asv, self.mol_attend(self.dropout(h_s_ex))).transpose(0,1), \
                           batch, dim_size=superatom_num).transpose(0,1)
        
        cs_i = F.elu(cs_i)

        #cs_i = F.elu(torch.mul(asv, self.mol_attend(self.dropout(h_s)))) # code 7 
        if self.debug:
            print('3 mol_context' ,cs_i.shape)
            
        # code 8
        return self.rnn(cs_i, h_s)    
    

class AtomEmbedding(torch.nn.Module):
    def __init__(self, atom_dim,  edge_dim, fp_dim, R=2, dropout = 0.2, debug=False):
        super(AtomEmbedding, self).__init__()
        self.R = R
        self.debug = debug
        self.conv = torch.nn.ModuleList([GatConvAtom(atom_in_channels=atom_dim, bond_in_channels= edge_dim, fingerprint_dim=fp_dim, dropout = dropout, debug=debug, step = i) for i in range(self.R)])  # GraphMultiHeadAttention

    def forward(self, x, edge_index, edge_attr):
        for i in range(self.R):
            if self.debug:
                print(x.shape)
            
            x = self.conv[i](x, edge_index, edge_attr) # code 1-12
            if self.debug:
                print(x.shape)    
        return x
    

class MoleculeEmbedding(torch.nn.Module):
    def __init__(self, fp_dim, dropout, debug, T=2):
        super(MoleculeEmbedding, self).__init__()
        self.T = T
        self.debug = debug
        self.conv =torch.nn.ModuleList([GatSuperNode(fp_dim, dropout, debug, step = i) for i in range(self.T)])

    def forward(self, h_s, x, edge_index):
        for i in range(self.T):
            h_s = self.conv[i](h_s, x, edge_index) # code 1-7
        return h_s

class AttentiveFP(torch.nn.Module):
    def __init__(self, atom_in_dim, edge_in_dim, fingerprint_dim=200, R=2, T=2, dropout=0.2,  debug = False, outdim=1):
        super(AttentiveFP, self).__init__()
        self.R = R
        self.T = T
        self.debug = debug
        self.dropout = dropout
        # call the atom embedding Phase
        self.convsAtom = AtomEmbedding(atom_in_dim, edge_in_dim, fingerprint_dim, R, debug) 
        self.convsMol = MoleculeEmbedding(fingerprint_dim, dropout, debug, T )

        # fast down project could be much more sofisticated! (ie  Feed Forward Network with multiple layers )
        self.out = Linear(fingerprint_dim, outdim) 
        
    def forward(self, data):
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
        x = F.dropout(self.convsAtom(x, edge_index, edge_attr), p=self.dropout, training=self.training) # atom Embedding       
        # molecular supernode definition / get init superatom by sum
        h_s = scatter_add(x.transpose(0,1), batch, dim_size=batch.max()+1).transpose(0,1)
        # we need to expend using the batch
        h_s = F.dropout(self.convsMol(h_s, x, batch), p=self.dropout, training=self.training) # molecule / supernode Embedding
        h_s = self.out(h_s)
        return h_s
