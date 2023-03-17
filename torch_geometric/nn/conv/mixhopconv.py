import torch
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class MixHopConv(MessagePassing):
    def __init__(self,
                 in_dim,
                 out_dim,
                 p=[0, 1, 2],
                 dropout=0,
                 activation=None,
                 batchnorm=False,kwargs):
      
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.p = p
        self.activation = activation
        self.batchnorm = batchnorm

        # define dropout layer
        self.dropout = nn.Dropout(dropout)

        # define batch norm layer
        if self.batchnorm:
            self.bn = nn.BatchNorm1d(out_dim * len(p))
        
        # define weight dict for each power j
        self.weights = nn.ModuleDict({
            str(j): nn.Linear(in_dim, out_dim, bias=False) for j in p
        })

        self.max_j =  max(self.p) + 1

    def forward(self, x, edge_index):

        outputs = []

        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        feats = x


        for j in range(self.max_j):

            if j in self.p:
                output = self.weights[str(j)](feats)
                outputs.append(output)

            
            feats = self.propagate(edge_index, x=x, norm=norm)

        final = torch.cat(outputs, dim=1)

        if self.batchnorm:
            final = self.bn(final)
        
        if self.activation is not None:
            final = self.activation(final)
        
        final = self.dropout(final)

        return final

    def message(self, x_j, norm):

        return norm.view(-1, 1) * x_j




class MixHop(nn.Module):
    def __init__(self,
                 in_dim,
                 hid_dim, 
                 out_dim,
                 num_layers=2,
                 p=[0, 1, 2],
                 input_dropout=0.0,
                 layer_dropout=0.0,
                 activation=None,
                 batchnorm=False):
        super(MixHop, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.p = p
        self.input_dropout = input_dropout
        self.layer_dropout = layer_dropout
        self.activation = activation
        self.batchnorm = batchnorm

        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(self.input_dropout)

        # Input layer
        self.layers.append(MixHopConv(self.in_dim,
                                      self.hid_dim,
                                      p=self.p,
                                      dropout=self.input_dropout,
                                      activation=self.activation,
                                      batchnorm=self.batchnorm))
        
        # Hidden layers with n - 1 MixHopConv layers
        for i in range(self.num_layers - 2):
            self.layers.append(MixHopConv(self.hid_dim * len(p),
                                          self.hid_dim,
                                          p=self.p,
                                          dropout=self.layer_dropout,
                                          activation=self.activation,
                                          batchnorm=self.batchnorm))
        
        self.fc_layers = nn.Linear(self.hid_dim * len(p), self.out_dim, bias=False)

    def forward(self, x,edge_index):
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, edge_index)
        
        x = self.fc_layers(x)

        return x

