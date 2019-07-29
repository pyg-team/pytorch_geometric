import torch
import torch.nn as nn

import torch_geometric
from torch_geometric.nn import TopKPooling, GCNConv
import torch_sparse


class GraphUNet(nn.Module):
    def __init__(self, n_channels, Conv=GCNConv, n_layers=5,
                 Activation=nn.ReLU, Normalization=nn.Identity,
                 Pool=TopKPooling, is_double_conv=False, max_nchannels=1024,
                 is_sum_res=True, factor_chan=2, **kwargs):
        super(GraphUNet, self).__init__()

        self.activation_ = Activation(inplace=True)
        self.is_double_conv = is_double_conv
        self.max_nchannels = max_nchannels
        self.is_sum_res = is_sum_res
        self.factor_chan = factor_chan
        self.n_layers = n_layers

        self.in_out_channels = self._get_in_out_channels(n_channels, n_layers)
        self.convs = nn.ModuleList([
            Conv(in_chan, out_chan, **kwargs)
            for in_chan, out_chan in self.in_out_channels
        ])

        self.norms = nn.ModuleList(
            [Normalization(out_chan) for _, out_chan in self.in_out_channels])

        pool_in_chan = [
            min(self.factor_chan**i * n_channels, max_nchannels)
            for i in range(n_layers // 2 + 1)
        ][1:]
        self.pools = nn.ModuleList([Pool(in_chan) for in_chan in pool_in_chan])

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        edge_weight = x.new_ones((edge_index.size(1), ))

        n_blocks = self.n_layers // 2 if self.is_double_conv else self.n_layers
        n_down_blocks = n_blocks // 2
        residuals = [None] * n_down_blocks
        edges = [None] * n_down_blocks
        perms = [None] * n_down_blocks

        # Down
        for i in range(n_down_blocks):
            x = self._apply_conv_block_i(x, edge_index, i,
                                         edge_weight=edge_weight)
            residuals[i] = x

            # not clear whether to save before or after augment
            edges[i] = (edge_index, edge_weight)

            # (A + I)^2
            edge_index, edge_weight = self.augment_adj(edge_index, edge_weight,
                                                       x.size(0))

            x, edge_index, edge_weight, batch, perm = self.pools[i](
                x, edge_index, edge_attr=edge_weight, batch=batch)
            perms[i] = perm

        # Bottleneck
        x = self._apply_conv_block_i(x, edge_index, n_down_blocks,
                                     edge_weight=edge_weight)

        # Up
        for i in range(n_down_blocks + 1, n_blocks):
            edge_index, edge_weight = edges[n_down_blocks - i]
            res = residuals[n_down_blocks - i]
            up = torch.zeros_like(res)
            up[perms[n_down_blocks - i]] = x
            if not self.is_sum_res:
                x = torch.cat((res, up), dim=1)  # conncat on channels
            else:
                x = res + up
            x = self._apply_conv_block_i(x, edge_index, i,
                                         edge_weight=edge_weight)

        return x

    def augment_adj(self, edge_index, edge_weight, n_nodes):
        edge_index, edge_weight = torch_geometric.utils.add_self_loops(
            edge_index, edge_weight=edge_weight)
        edge_index, edge_weight = torch_geometric.utils.sort_edge_index(
            edge_index, edge_weight, n_nodes)
        edge_index, edge_weight = torch_sparse.spspmm(
            edge_index, edge_weight, edge_index, edge_weight, n_nodes, n_nodes,
            n_nodes)
        return edge_index, edge_weight

    def _apply_conv_block_i(self, x, edge_index, i, **kwargs):
        """Apply the i^th convolution block."""

        x = self.activation_(self.norms[i](self.convs[i](x, edge_index,
                                                         **kwargs)))
        return x

    def _get_in_out_channels(self, n_channels, n_layers):
        """Return a list of tuple of input and output channels for a Unet."""

        if self.is_double_conv:
            assert n_layers % 2 == 0, "n_layers={} not even".format(n_layers)
            # e.g. if n_channels=16, n_layers=10: [16, 32, 64]
            channel_list = [
                self.factor_chan**i * n_channels
                for i in range(n_layers // 4 + 1)
            ]
            # e.g.: [16, 16, 32, 32, 64, 64]
            channel_list = [i for i in channel_list for _ in (0, 1)]
            # e.g.: [16, 16, 32, 32, 64, 64, 64, 32, 32, 16, 16]
            channel_list = channel_list + channel_list[-2::-1]
            # bound max number of channels by self.max_nchannels
            channel_list = [min(c, self.max_nchannels) for c in channel_list]
            # e.g.: [..., (32, 32), (32, 64), (64, 64), (64, 32), (32, 32), (32, 16) ...]
            in_out_channels = self._list_chan_to_tuple(channel_list, n_layers)
            if not self.is_sum_res:
                # e.g.: [..., (32, 32), (32, 64), (64, 64), (128, 32), (32, 32), (64, 16) ...]
                # due to concat
                idcs = slice(
                    len(in_out_channels) // 2 + 1, len(in_out_channels), 2)
                in_out_channels[idcs] = [
                    (in_chan * 2, out_chan)
                    for in_chan, out_chan in in_out_channels[idcs]
                ]
        else:
            assert n_layers % 2 == 1, "n_layers={} not odd".format(n_layers)
            # e.g. if n_channels=16, n_layers=5: [16, 32, 64]
            channel_list = [
                self.factor_chan**i * n_channels
                for i in range(n_layers // 2 + 1)
            ]
            # e.g.: [16, 32, 64, 64, 32, 16]
            channel_list = channel_list + channel_list[::-1]
            # bound max number of channels by self.max_nchannels
            channel_list = [min(c, self.max_nchannels) for c in channel_list]
            # e.g.: [(16, 32), (32,64), (64, 64), (64, 32), (32, 16)]
            in_out_channels = self._list_chan_to_tuple(channel_list, n_layers)
            if not self.is_sum_res:
                # e.g.: [(16, 32), (32,64), (64, 64), (128, 32), (64, 16)] due to concat
                idcs = slice(
                    len(in_out_channels) // 2 + 1, len(in_out_channels))
                in_out_channels[idcs] = [
                    (in_chan * 2, out_chan)
                    for in_chan, out_chan in in_out_channels[idcs]
                ]

        return in_out_channels

    def _list_chan_to_tuple(self, n_channels, n_layers):
        """Return a list of tuple of input and output channels."""
        channel_list = list(n_channels)
        assert len(channel_list) == n_layers + 1
        return list(zip(channel_list, channel_list[1:]))
