import torch
import torch.nn.functional as F
from torch.nn import Linear
from tqdm import tqdm

from torch_geometric.nn import GATConv


class GATBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads, last_layer=False,
                 **conv_kwargs):
        super().__init__()

        self.conv = GATConv(in_channels, out_channels, heads, **conv_kwargs)
        self.skip = Linear(
            in_channels, out_channels if last_layer else out_channels * heads)
        self.last_layer = last_layer

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        # TODO: how to use skip connection with NeighborLoader?
        # x = x + self.skip(?)
        return x if self.last_layer else F.elu(x)


class GATNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads,
                 num_layers):
        super().__init__()

        self.layers = torch.nn.ModuleList()
        self.layers.append(GATBlock(in_channels, hidden_channels, heads))
        for _ in range(num_layers - 2):
            self.layers.append(
                GATBlock(hidden_channels * heads, hidden_channels, heads))
        self.layers.append(
            GATBlock(hidden_channels * heads, out_channels, heads,
                     last_layer=True, concat=False))

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index)
        return x

    @torch.no_grad()
    def inference(self, subgraph_loader, device):
        for batch in tqdm(subgraph_loader):
            batch = batch.to(device)
            batch_size = batch.batch_size
            out = self(batch.x, batch.edge_index)[:batch_size]
