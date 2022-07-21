import torch
from tqdm import tqdm

from torch_geometric.nn import SAGEConv, to_hetero


class HeteroGraphSAGE(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, num_layers, output_channels):
        super().__init__()
        self.model = to_hetero(
            SAGEForHetero(hidden_channels, num_layers, output_channels),
            metadata)  # TODO: replace by basic_gnn

    @torch.no_grad()
    def inference(self, loader, device, progress_bar=False):
        self.model.eval()
        if progress_bar:
            loader = tqdm(loader, desc="Inference")
        for batch in loader:
            batch = batch.to(device)
            self.model(batch.x_dict, batch.edge_index_dict)


class SAGEForHetero(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, out_channels):
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
