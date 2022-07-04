import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.nn import EdgeConv
from torch.nn import Linear as Lin
from torch.nn import ReLU
from torch.nn import Sequential as Seq


class EdgeConvNet(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, out_channels,
                 num_layers):
        super().__init__()
        nn_in = Seq(Lin(2 * input_channels, hidden_channels), ReLU(),
                    Lin(hidden_channels, hidden_channels))
        nn_hid = Seq(Lin(2 * hidden_channels, hidden_channels), ReLU(),
                     Lin(hidden_channels, hidden_channels))
        nn_out = Seq(Lin(2 * hidden_channels, hidden_channels), ReLU(),
                     Lin(hidden_channels, out_channels))
        self.convs = torch.nn.ModuleList()
        self.convs.append(EdgeConv(nn_in))
        for _ in range(num_layers - 2):
            self.convs.append(EdgeConv(nn_hid))
        self.convs.append(EdgeConv(nn_out))

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
