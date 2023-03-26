from functools import reduce
from operator import mul

import torch
from torch import nn
from torch.nn import functional as F

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import DenseGCNConv, GCNConv


class GraphWaveNet(nn.Module):
    r""" GraphWaveNet implementation from the
    `Graph WaveNet for Deep Spatial-Temporal Graph Modeling
    <https://arxiv.org/abs/1906.00121>`_ paper. The model takes a list of
    node embeddings across several time steps and predicts the output
    embeddings for the specified number of upcoming time steps.

    .. note::
        For examples of the GraphWaveNet layer, see
        `examples/contrib/graphwavenet/main.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        contrib/graphwavenet/main.py>`_ .

    Args:
        num_nodes (int): Number of nodes in the input graph.
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        out_timesteps (int): The number of time steps for which predictions
            are generated.
        dilations (list(int), optional): The set of dilations applied by each
            temporal convolution layer. For each dilation :obj:`d`, the
            temporal convolution captures information from :obj:`d` time steps
            prior. (default: :obj:`[1, 2, 1, 2, 1, 2, 1, 2]`)
        adaptive_embeddings (int, optional): The size of the embeddings used
            to calculate the adaptive matrix. (default: :obj:`10`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.3`)
        residual_channels (int, optional): Number of residual channels
            (default: :obj:`32`)
        dilation_channels (int, optional): Number of dilation channels
            (default: :obj:`32`)
        skip_channels (int, optional): Number of skip layer channels
            (default: :obj:`256`)
        end_channels (int, optional): Size of the final linear layer
            (default: :obj:`512`)

    """
    def __init__(self, num_nodes, in_channels, out_channels, out_timesteps,
                 dilations=[1, 2, 1, 2, 1, 2, 1, 2], adptive_embeddings=10,
                 dropout=0.3, residual_channels=32, dilation_channels=32,
                 skip_channels=256, end_channels=512):

        super(GraphWaveNet, self).__init__()

        self.total_dilation = sum(dilations)
        self.num_dilations = len(dilations)
        self.num_nodes = num_nodes

        self.dropout = dropout

        self.e1 = nn.Parameter(torch.randn(num_nodes, adptive_embeddings),
                               requires_grad=True)
        self.e2 = nn.Parameter(torch.randn(adptive_embeddings, num_nodes),
                               requires_grad=True)

        self.input = nn.Conv2d(in_channels=in_channels,
                               out_channels=residual_channels,
                               kernel_size=(1, 1))

        self.tcn_a = nn.ModuleList()
        self.tcn_b = nn.ModuleList()
        self.gcn = nn.ModuleList()
        self.gcn_adp = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.skip = nn.ModuleList()

        self.out_timesteps = out_timesteps
        self.out_channels = out_channels

        for d in dilations:
            self.tcn_a.append(
                nn.Conv2d(in_channels=residual_channels,
                          out_channels=dilation_channels, kernel_size=(1, 2),
                          dilation=d))

            self.tcn_b.append(
                nn.Conv2d(in_channels=residual_channels,
                          out_channels=dilation_channels, kernel_size=(1, 2),
                          dilation=d))

            # GCNConv is used for performing graph convolutions over the
            # normal adjacency matrix
            self.gcn.append(
                GCNConv(in_channels=dilation_channels,
                        out_channels=residual_channels))

            # Since the adaptive matrix is a softmax output, it represents a
            # dense graph
            # For fast training and inference, we use a DenseGCNConv layer
            self.gcn_adp.append(
                DenseGCNConv(in_channels=dilation_channels,
                             out_channels=residual_channels))

            self.skip.append(
                nn.Conv2d(in_channels=residual_channels,
                          out_channels=skip_channels, kernel_size=(1, 1)))
            self.bn.append(nn.BatchNorm2d(residual_channels))

        self.end1 = nn.Conv2d(in_channels=skip_channels,
                              out_channels=end_channels, kernel_size=(1, 1))
        self.end2 = nn.Conv2d(in_channels=end_channels,
                              out_channels=out_channels * out_timesteps,
                              kernel_size=(1, 1))

    def forward(self, x, edge_index, edge_weight=None):
        r"""
        The forward pass for the GraphWaveNet
        
        Args:
            x (torch.Tensor): The temporal node features of shape
                :math:`(*, t_{input}, |\mathcal{V}|, F_{in})`
            edge_index (torch.Tensor): The edge indices 
                of shape :math:`(2, |\mathcal{E}|)`
            edge_weight (torch.Tensor, optional): The edge weights 
                shape of :math:`(|\mathcal{E}|)` (default: :obj:`None`)

        Return types:
            * **x** (*torch.Tensor*): Predicted temporal node features of shape 
              :math:`(*, t_{output}, |\mathcal{V}|, F_{out})`
        """

        # Both batched and unbatched input is supported
        # In the case of a single batch, the input is broadcast
        is_batched = True
        if len(x.size()) == 3:
            is_batched = False
            x = torch.unsqueeze(x, dim=0)

        batch_size = x.size()[:-3]
        input_timesteps = x.size(-3)

        x = x.transpose(-1, -3)

        # If the dilation steps require a larger number of input time steps,
        # padding is performed
        if self.total_dilation + 1 > input_timesteps:
            x = F.pad(x, (self.total_dilation - input_timesteps + 1, 0))

        x = self.input(x)

        # The adaptive adjacency matrix is the product of 2 learnable
        # embedding matrices
        # ReLU is used toeliminate weak connections
        # SoftMax function is applied to normalize the self-adaptive adjacency
        # matrix
        adp = F.softmax(F.relu(torch.mm(self.e1, self.e2)), dim=1)

        skip_out = None

        for k in range(self.num_dilations):
            residual = x

            # TCN layer
            g1 = self.tcn_a[k](x)
            g1 = torch.tanh(g1)

            g2 = self.tcn_b[k](x)
            g2 = torch.sigmoid(g2)

            g = g1 * g2

            # The skip connection output is aggregated before the GCN layer
            skip_cur = self.skip[k](g)

            if skip_out is None:
                skip_out = skip_cur
            else:
                # Since dilation reduces the number of time steps,
                # only the required number of latest time steps are considered
                skip_out = skip_out[..., -skip_cur.size(-1):] + skip_cur

            g = g.transpose(-1, -3)

            timesteps = g.size(-3)
            g = g.reshape(reduce(mul, g.size()[:-2]), *g.size()[-2:])

            # The data for several time steps in batched into a single batch
            # This helps speed up the model by passing a single large batch to
            # the GCN
            data = self.__batch_timesteps__(g, edge_index,
                                            edge_weight).to(g.device)

            # GCN layer
            # One GCN is for the actual adjacency matrix
            # The other GCN is for the adaptive matrix
            gcn_out = self.gcn[k](data.x, data.edge_index,
                                  edge_weight=torch.flatten(data.edge_attr))
            gcn_out = gcn_out.reshape(*batch_size, timesteps, -1,
                                      self.gcn[k].out_channels)

            gcn_out_adp = self.gcn_adp[k](g, adp)
            gcn_out_adp = gcn_out_adp.reshape(*batch_size, timesteps, -1,
                                              self.gcn[k].out_channels)

            x = gcn_out + gcn_out_adp

            x = F.dropout(x, p=self.dropout)

            x = x.transpose(-3, -1)

            # The residual connection is fed to the next spatial-temporal layer
            x = x + residual[..., -x.size(-1):]
            x = self.bn[k](x)

        skip_out = skip_out[..., -1:]

        x = torch.relu(skip_out)

        # Final linear layer
        x = self.end1(x)
        x = torch.relu(x)

        # Transforming the output to the appropriate number of channels
        # (out_timesteps * self.out_channels)
        x = self.end2(x)

        # The output is reshaped into the expected final shape
        if is_batched:
            x = x.reshape(*batch_size, self.out_timesteps, self.out_channels,
                          self.num_nodes).transpose(-1, -2)
        else:
            x = x.reshape(self.out_timesteps, self.out_channels,
                          self.num_nodes).transpose(-1, -2)

        return x

    def __batch_timesteps__(self, x, edge_index, edge_weight=None):
        r"""
        This method batches the data for several time steps into a single batch
        """
        edge_index = edge_index.expand(x.size(0), *edge_index.shape)

        if edge_weight is not None:
            edge_weight = edge_weight.expand(x.size(0), *edge_weight.shape)

        dataset = [
            Data(x=_x, edge_index=e_i, edge_attr=e_w)
            for _x, e_i, e_w in zip(x, edge_index, edge_weight)
        ]
        loader = DataLoader(dataset=dataset, batch_size=x.size(0))

        return next(iter(loader))
