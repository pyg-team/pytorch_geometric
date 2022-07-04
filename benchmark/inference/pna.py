import torch
from tqdm import tqdm
from torch_geometric.nn import PNAConv


class PNANet(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, out_channels,
                 num_layers, degree):
        super().__init__()
        self.aggregators = ['mean', 'min', 'max', 'std']
        self.scalers = ['identity', 'amplification', 'attenuation']
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            PNAConv(input_channels, hidden_channels, self.aggregators,
                    self.scalers, degree))
        for i in range(num_layers - 2):
            self.convs.append(
                PNAConv(hidden_channels, hidden_channels, self.aggregators,
                        self.scalers, degree))
        self.convs.append(
            PNAConv(hidden_channels, out_channels, self.aggregators,
                    self.scalers, degree))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu_()
        return x

    @torch.no_grad()
    def inference(self, subgraph_loader, device):
        for batch in tqdm(subgraph_loader):
            batch = batch.to(device)
            batch_size = batch.batch_size
            out = self(batch.x, batch.edge_index)[:batch_size]
