import torch
from tqdm import tqdm

from torch_geometric.nn import SAGEConv, to_hetero


class SAGE_HETERO:
    def __init__(self, hidden_channels, output_channels, num_layers) -> None:
        self.model = None
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.num_layers = num_layers

    def create_hetero(self, metadata):
        model = SAGE_FOR_HETERO(self.hidden_channels, self.output_channels,
                                self.num_layers)
        self.model = to_hetero(model, metadata, aggr='sum')

    def inference(self, loader, device):
        self.model.eval()
        for batch in tqdm(loader):
            batch = batch.to(device)
            batch_size = batch['paper'].batch_size
            out = self.model(batch.x_dict,
                             batch.edge_index_dict)['paper'][:batch_size]


class SAGE_FOR_HETERO(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv((-1, -1), hidden_channels))
        for i in range(num_layers - 2):
            self.convs.append(SAGEConv((-1, -1), hidden_channels))
        self.convs.append(SAGEConv((-1, -1), out_channels))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu_()
        return x
