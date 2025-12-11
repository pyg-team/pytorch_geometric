import torch
from torch.nn import Linear, ReLU
from torch_geometric.nn.conv import MessagePassing

class GNCAConv(MessagePassing):
    r"""The Graph Neural Cellular Automata convolution layer from the
    `"Learning Graph Cellular Automata" <https://arxiv.org/abs/2110.14237>`_ paper,
    with optional edge features..

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{x}_i \, \| \, \sum_{j \in \mathcal{N}(i)}
        \mathrm{ReLU}(\mathbf{W} \mathbf{x}_j + \mathbf{b})

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample (message dimension).
        edge_dim (int, optional): Size of each edge feature. (default: :obj:`0`)
        aggr (string, optional): The aggregation scheme to use (:obj:`"add"`,
            :obj:`"mean"`, :obj:`"max"`). (default: :obj:`"add"`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels, out_channels, 
                 edge_dim=0,
                 aggr='add', **kwargs):
        super().__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.edge_dim = edge_dim

        self.lin_node = Linear(in_channels, out_channels, bias=True)

        # If there are edge features, we need a separate linear layer for them
        if edge_dim > 0:
            # bias False because it's handled in lin_node
            self.lin_edge = Linear(edge_dim, out_channels, bias=False)
        else:
            self.register_parameter('lin_edge', None)
        
        self.act = ReLU()

        self.reset_parameters()

    def reset_parameters(self):
            self.lin_node.reset_parameters()
            if self.lin_edge is not None:
                self.lin_edge.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # 1. Transform node features
        x_proj = self.lin_node(x)

        # 2. Transform edge features (if they exist)
        edge_emb = torch.empty(0, device=x.device, dtype=x.dtype)
        if self.lin_edge is not None and edge_attr is not None:
            edge_emb = self.lin_edge(edge_attr)

        # 3. Propagate messages (pass in transformed node and edge features) 
        out = self.propagate(edge_index, x=x_proj, edge_emb=edge_emb)
        
        # 4. Concatenate with self (h_i || sum(...))
        return torch.cat([x, out], dim=-1)

    def message(self, x_j, edge_emb):
        # x_j has shape [E, in_channels]

        # Start with the transformed node features
        msg = x_j

        # If there are edge features, add them to the message
        if edge_emb.numel() > 0:
            msg = msg + edge_emb 
        
        # Apply the activation
        return self.act(msg)

    def __repr__(self):
            return (f'{self.__class__.__name__}({self.in_channels}, '
                    f'{self.out_channels}, edge_dim={self.edge_dim})')