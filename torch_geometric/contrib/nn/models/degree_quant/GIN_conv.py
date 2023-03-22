import torch
from torch_geometric.utils import remove_self_loops


class GINConvQuant(MessagePassingQuant):
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
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_index, _ = remove_self_loops(edge_index)
        out = self.propagate(edge_index, x=x, mask=mask)
        return out

    def message(self, x_j):
        return x_j

    def update(self, aggr_out, x):
        return self.nn((1 + self.eps) * x + aggr_out)

    def __repr__(self):
        return "{}(nn={})".format(self.__class__.__name__, self.nn)