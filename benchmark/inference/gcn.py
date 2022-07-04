import torch
import torch.nn.functional as F
from tqdm import tqdm

from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = self.convs[-1](x, edge_index)
        return x

    @torch.no_grad()
    def inference(self, subgraph_loader, device):
        for batch in tqdm(subgraph_loader):
            batch = batch.to(device)
            batch_size = batch.batch_size
            out = self(batch.x, batch.edge_index)[:batch_size]
