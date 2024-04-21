import math
from typing import Literal, Tuple, List

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.aggr import MaxAggregation, SumAggregation


class CMPNN(nn.Module):
    r"""The Communicative Message Passing Neural Network (CMPNN) model from the
    `"Communicative Representation Learning on Attributed Molecular Graphs"
    <https://www.ijcai.org/Proceedings/2020/392>`_ paper.

    .. note::

        For an example of using CMPNN, see
        `examples/cmpnn.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        cmpnn.py>`_.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Hidden node feature dimensionality.
        out_channels (int): Size of each output sample.
        edge_dim (int): Edge feature dimensionality.
        num_layers (int): Number of GNN layers.
        communicator (Literal['additive', 'inner_product', 'gru', 'mlp']): Name
            of the communicative kernel used for message interactions between
            nodes and edges.
        dropout (float, optional): Dropout probability. (default: :obj:`0.0`)
        bias (float, optional): Bias. (default: :obj:`False`)
    """
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, edge_dim: int, num_layers: int,
                 communicator: Literal['additive', 'inner_product', 'gru',
                                       'mlp'], dropout: float = 0.0,
                 bias: bool = False) -> None:

        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.communicator = communicator
        self.dropout = dropout

        self.atom_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_channels, bias=bias), nn.ReLU())
        self.bond_proj = nn.Sequential(
            nn.Linear(edge_dim, hidden_channels, bias=bias), nn.ReLU())

        self.convs = nn.ModuleList()
        for _ in range(self.num_layers - 1):
            self.convs.append(
                GCNEConv(hidden_channels, hidden_channels, communicator,
                         dropout, bias))
        self.convs.append(
            GCNConv(hidden_channels, hidden_channels, communicator))

        self.lin = nn.Linear(hidden_channels * 3, hidden_channels, bias=bias)
        self.gru = BatchGRU(hidden_channels)
        self.seq_out = nn.Sequential(
            nn.Linear(hidden_channels * 2, out_channels), nn.ReLU(),
            nn.Dropout(p=dropout))

        self.apply(self._init_weights)

    def _init_weights(self, module) -> None:

        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor,
                batch: Tensor) -> Tensor:
        r"""Forward pass.

        Args:
            x (torch.Tensor): The node features.
            edge_index (torch.Tensor): The edge indices.
            edge_attr (torch.Tensor): The edge features.
            batch (torch.Tensor): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each node to a specific graph.
        """

        init_atom_embed = self.atom_proj(x)
        atom_embed = init_atom_embed.clone()
        init_bond_embed = self.bond_proj(edge_attr)
        bond_embed = init_bond_embed.clone()

        for i, layer in enumerate(self.convs):
            if i == len(self.convs) - 1:
                aggr_message = layer(atom_embed, bond_embed, edge_index)
            else:
                atom_embed, bond_embed = layer(atom_embed, bond_embed,
                                               edge_index, init_bond_embed)

        # `aggr_message`: Message from incoming bonds
        # `atom_embed`: Current atom's representation
        # `init_atom_embed`: Atom's initial representation
        atom_embed = self.lin(
            torch.cat([aggr_message, atom_embed, init_atom_embed], 1))
        atom_embed = self.gru(atom_embed, batch)
        return self.seq_out(atom_embed)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'in_channels={self.in_channels}, '
                f'hidden_channels={self.hidden_channels}, '
                f'out_channels={self.out_channels}, '
                f'edge_dim={self.edge_dim}, '
                f'num_layers={self.num_layers}, '
                f'communicator={self.communicator}'
                f')')


class GCNEConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int,
                 communicator: Literal['additive', 'inner_product', 'gru',
                                       'mlp'], dropout: float = 0.0,
                 bias: bool = False) -> None:

        super().__init__(
            aggr=[SumAggregation(), MaxAggregation()],
            aggr_kwargs=dict(mode='message_booster'),
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.communicator = communicator
        self.dropout = dropout
        self.communicator = NodeEdgeMessageCommunicator(
            name=communicator, hidden_channels=in_channels)
        self.lin = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x: Tensor, edge_attr: Tensor, edge_index: Tensor,
                init_edge_embed: Tensor) -> Tuple[Tensor, Tensor]:

        x = self.propagate(
            edge_index,
            x=x,
            edge_attr=edge_attr,
            # Aggregation is done on the `edge_attr` based on
            # index(`edge_index_i`).
            # The output of the aggregation should be of the
            # shape (num_atoms x hidden_channels).
            # To have this output, we pass the expected `dim_size` in
            # the `size` variable.
            # In the `_collect` method, `x.size(0)` will be assigned
            # to `dim_size` and will get passed to the `aggregate` method.
            size=[x.size(0), None])
        edge_attr = self.edge_updater(edge_index, x=x, edge_attr=edge_attr,
                                      init_edge_embed=init_edge_embed)
        return x, edge_attr

    def message(self, edge_attr: Tensor) -> Tensor:
        return edge_attr

    def update(self, message: Tensor, x: Tensor) -> Tensor:
        return self.communicator(message, x)

    def edge_update(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor,
                    init_edge_embed: Tensor) -> Tensor:

        rev_edge_index = GCNEConv.get_reverse_edge_index(
            num_edges=edge_attr.size(0))
        rev_edge_index = torch.tensor(rev_edge_index, device=edge_attr.device,
                                      dtype=torch.long)
        # Avoid using `x_i` or `x_j` instead of `x`, since `x_*` depends on the
        # `self.flow` (source_to_target or target_to_source) and determines the
        # i and j values.
        edge_attr = x[edge_index[0]] - edge_attr[rev_edge_index]
        edge_attr = self.lin(edge_attr)
        edge_attr = F.dropout(F.relu(init_edge_embed + edge_attr),
                              p=self.dropout, training=self.training)
        return edge_attr

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}({self.in_channels}, '
            f"{self.out_channels}, communicator='{self.communicator}')")

    @staticmethod
    def get_reverse_edge_index(num_edges: int) -> List[int]:

        rev_edge_index = []
        for i in range(int(num_edges / 2)):
            rev_edge_index.extend([i * 2 + 1, i * 2])
        return rev_edge_index


class GCNConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        communicator: Literal['additive', 'inner_product', 'gru', 'mlp'],
    ) -> None:

        super().__init__(
            aggr=[SumAggregation(), MaxAggregation()],
            aggr_kwargs=dict(mode='message_booster'),
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.communicator = communicator
        self.communicator = NodeEdgeMessageCommunicator(
            name=communicator, hidden_channels=in_channels)

    def forward(self, x: Tensor, edge_attr: Tensor,
                edge_index: Tensor) -> Tuple[Tensor, Tensor]:

        x = self.propagate(
            edge_index,
            x=x,
            edge_attr=edge_attr,
            # Aggregation is done on the `edge_attr` based on
            # index(`edge_index_i`).
            # The output of the aggregation should be of the
            # shape (num_atoms x hidden_channels).
            # To have this output, we pass the expected `dim_size` in
            # the `size` variable.
            # In the `_collect` method, `x.size(0)` will be assigned
            # to `dim_size` and will get passed to the `aggregate` method.
            size=[x.size(0), None])
        return x

    def message(self, edge_attr: Tensor) -> Tensor:
        return edge_attr

    def update(self, message: Tensor, x: Tensor) -> Tensor:
        return self.communicator(message, x)

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}({self.in_channels}, '
            f"{self.out_channels}, communicator='{self.communicator}')")


class NodeEdgeMessageCommunicator(nn.Module):
    def __init__(self, name: Literal['additive', 'inner_product', 'gru',
                                     'mlp'], hidden_channels: int) -> None:

        super().__init__()
        assert name in ('additive', 'inner_product', 'gru',
                        'mlp'), f"Invalid communicator '{name}'!"
        self.name = name
        self.hidden_channels = hidden_channels
        self.communicator = None

        if name == 'gru':
            self.communicator = nn.GRUCell(hidden_channels, hidden_channels)
        elif name == 'mlp':
            self.communicator = nn.Sequential(
                nn.Linear(hidden_channels * 2, hidden_channels), nn.ReLU())

    def forward(self, message: Tensor, hidden_state: Tensor) -> Tensor:

        if self.name == 'additive':
            out = hidden_state + message
        elif self.name == 'inner_product':
            out = hidden_state * message
        elif self.name == 'gru':
            out = self.communicator(hidden_state, message)
        elif self.name == 'mlp':
            message = torch.cat((hidden_state, message), dim=1)
            out = self.communicator(message)
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.hidden_channels}, '
                f"name='{self.name}')")


class BatchGRU(nn.Module):
    def __init__(self, hidden_channels: int, num_layers: int = 1) -> None:

        super().__init__()
        self.hidden_channels = hidden_channels
        self.gru = nn.GRU(hidden_channels, hidden_channels,
                          num_layers=num_layers, batch_first=True,
                          bidirectional=True)
        self.bias = nn.Parameter(torch.Tensor(hidden_channels))
        self.bias.data.uniform_(-1.0 / math.sqrt(self.hidden_channels),
                                1.0 / math.sqrt(self.hidden_channels))

    def forward(self, h_atom: Tensor, batch: Tensor) -> Tensor:

        device = h_atom.device
        num_atoms = h_atom.shape[0]
        message = F.relu(h_atom + self.bias)
        unique_values, counts = torch.unique(batch, return_counts=True)
        dim_1 = unique_values.shape[
            0]  # No. of mol graphs in the batch (aka batch size)
        dim_2 = counts.max().item()  # Maximum no. of atoms in the batch
        dim_3 = self.hidden_channels

        messages = torch.zeros((dim_1, dim_2, dim_3), device=device)
        hidden_states = torch.zeros((2, dim_1, dim_3),
                                    device=device)  # 2 -> bidirectional
        for i, value in enumerate(unique_values):
            indices = (batch == value).nonzero().squeeze(1)
            num_samples = counts[i]
            messages[i, :num_samples] = message[indices]
            hidden_states[:, i, :] = h_atom[indices].max(0)[0]

        h_messages, _ = self.gru(messages, hidden_states)

        unpadded_messages = torch.zeros((num_atoms, dim_3 * 2), device=device)
        for i, value in enumerate(unique_values):
            num_samples = counts[i]
            unpadded_messages[batch == value, :] = h_messages[
                i, :num_samples].view(-1, dim_3 * 2)

        return unpadded_messages
