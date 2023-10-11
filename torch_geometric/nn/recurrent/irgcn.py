import torch
from torch.nn import Parameter
from torch_geometric.nn import RGCNConv
from torch_geometric.nn.inits import glorot, zeros


class LRGCN(torch.nn.Module):
    r"""An implementation of the Long Short Term Memory Relational
    Graph Convolution Layer. For details see this paper: `"Predicting Path
    Failure In Time-Evolving Graphs." <https://arxiv.org/abs/1905.03994>`_

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        num_relations (int): Number of relations.
        num_bases (int): Number of bases.
    """

    def __init__(
        self, in_channels: int, out_channels: int, num_relations: int, num_bases: int
    ):
        super(LRGCN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases
        self._create_layers()

    def _create_input_gate_layers(self):

        self.conv_x_i = RGCNConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            num_relations=self.num_relations,
            num_bases=self.num_bases,
        )

        self.conv_h_i = RGCNConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            num_relations=self.num_relations,
            num_bases=self.num_bases,
        )

    def _create_forget_gate_layers(self):

        self.conv_x_f = RGCNConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            num_relations=self.num_relations,
            num_bases=self.num_bases,
        )

        self.conv_h_f = RGCNConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            num_relations=self.num_relations,
            num_bases=self.num_bases,
        )

    def _create_cell_state_layers(self):

        self.conv_x_c = RGCNConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            num_relations=self.num_relations,
            num_bases=self.num_bases,
        )

        self.conv_h_c = RGCNConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            num_relations=self.num_relations,
            num_bases=self.num_bases,
        )

    def _create_output_gate_layers(self):

        self.conv_x_o = RGCNConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            num_relations=self.num_relations,
            num_bases=self.num_bases,
        )

        self.conv_h_o = RGCNConv(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            num_relations=self.num_relations,
            num_bases=self.num_bases,
        )

    def _create_layers(self):
        self._create_input_gate_layers()
        self._create_forget_gate_layers()
        self._create_cell_state_layers()
        self._create_output_gate_layers()

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return H

    def _set_cell_state(self, X, C):
        if C is None:
            C = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return C

    def _calculate_input_gate(self, X, edge_index, edge_type, H, C):
        I = self.conv_x_i(X, edge_index, edge_type)
        I = I + self.conv_h_i(H, edge_index, edge_type)
        I = torch.sigmoid(I)
        return I

    def _calculate_forget_gate(self, X, edge_index, edge_type, H, C):
        F = self.conv_x_f(X, edge_index, edge_type)
        F = F + self.conv_h_f(H, edge_index, edge_type)
        F = torch.sigmoid(F)
        return F

    def _calculate_cell_state(self, X, edge_index, edge_type, H, C, I, F):
        T = self.conv_x_c(X, edge_index, edge_type)
        T = T + self.conv_h_c(H, edge_index, edge_type)
        T = torch.tanh(T)
        C = F * C + I * T
        return C

    def _calculate_output_gate(self, X, edge_index, edge_type, H, C):
        O = self.conv_x_o(X, edge_index, edge_type)
        O = O + self.conv_h_o(H, edge_index, edge_type)
        O = torch.sigmoid(O)
        return O

    def _calculate_hidden_state(self, O, C):
        H = O * torch.tanh(C)
        return H

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_type: torch.LongTensor,
        H: torch.FloatTensor = None,
        C: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        """
        Making a forward pass. If the hidden state and cell state matrices are
        not present when the forward pass is called these are initialized with zeros.

        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node features.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_type** *(PyTorch Long Tensor)* - Edge type vector.
            * **H** *(PyTorch Float Tensor, optional)* - Hidden state matrix for all nodes.
            * **C** *(PyTorch Float Tensor, optional)* - Cell state matrix for all nodes.

        Return types:
            * **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
            * **C** *(PyTorch Float Tensor)* - Cell state matrix for all nodes.
        """
        H = self._set_hidden_state(X, H)
        C = self._set_cell_state(X, C)
        I = self._calculate_input_gate(X, edge_index, edge_type, H, C)
        F = self._calculate_forget_gate(X, edge_index, edge_type, H, C)
        C = self._calculate_cell_state(X, edge_index, edge_type, H, C, I, F)
        O = self._calculate_output_gate(X, edge_index, edge_type, H, C)
        H = self._calculate_hidden_state(O, C)
        return H, C
        