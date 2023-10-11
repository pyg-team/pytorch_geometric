import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class MPNNLSTM(nn.Module):
    r"""An implementation of the Message Passing Neural Network with Long Short Term Memory.
    For details see this paper: `"Transfer Graph Neural Networks for Pandemic Forecasting." <https://arxiv.org/abs/2009.08388>`_

    Args:
        in_channels (int): Number of input features.
        hidden_size (int): Dimension of hidden representations.
        num_nodes (int): Number of nodes in the network.
        window (int): Number of past samples included in the input.
        dropout (float): Dropout rate.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        num_nodes: int,
        window: int,
        dropout: float,
    ):
        super(MPNNLSTM, self).__init__()

        self.window = window
        self.num_nodes = num_nodes
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.in_channels = in_channels

        self._create_parameters_and_layers()

    def _create_parameters_and_layers(self):

        self._convolution_1 = GCNConv(self.in_channels, self.hidden_size)
        self._convolution_2 = GCNConv(self.hidden_size, self.hidden_size)

        self._batch_norm_1 = nn.BatchNorm1d(self.hidden_size)
        self._batch_norm_2 = nn.BatchNorm1d(self.hidden_size)

        self._recurrent_1 = nn.LSTM(2 * self.hidden_size, self.hidden_size, 1)
        self._recurrent_2 = nn.LSTM(self.hidden_size, self.hidden_size, 1)

    def _graph_convolution_1(self, X, edge_index, edge_weight):
        X = F.relu(self._convolution_1(X, edge_index, edge_weight))
        X = self._batch_norm_1(X)
        X = F.dropout(X, p=self.dropout, training=self.training)
        return X

    def _graph_convolution_2(self, X, edge_index, edge_weight):
        X = F.relu(self._convolution_2(X, edge_index, edge_weight))
        X = self._batch_norm_2(X)
        X = F.dropout(X, p=self.dropout, training=self.training)
        return X

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Making a forward pass through the whole architecture.

        Arg types:
            * **X** *(PyTorch FloatTensor)* - Node features.
            * **edge_index** *(PyTorch LongTensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch LongTensor, optional)* - Edge weight vector.

        Return types:
            *  **H** *(PyTorch FloatTensor)* - The hidden representation of size 2*nhid+in_channels+window-1 for each node.
        """
        R = list()

        S = X.view(-1, self.window, self.num_nodes, self.in_channels)
        S = torch.transpose(S, 1, 2)
        S = S.reshape(-1, self.window, self.in_channels)
        O = [S[:, 0, :]]

        for l in range(1, self.window):
            O.append(S[:, l, self.in_channels - 1].unsqueeze(1))

        S = torch.cat(O, dim=1)

        X = self._graph_convolution_1(X, edge_index, edge_weight)
        R.append(X)

        X = self._graph_convolution_2(X, edge_index, edge_weight)
        R.append(X)

        X = torch.cat(R, dim=1)

        X = X.view(-1, self.window, self.num_nodes, X.size(1))
        X = torch.transpose(X, 0, 1)
        X = X.contiguous().view(self.window, -1, X.size(3))

        X, (H_1, _) = self._recurrent_1(X)
        X, (H_2, _) = self._recurrent_2(X)

        H = torch.cat([H_1[0, :, :], H_2[0, :, :], S], dim=1)
        return H
        