import torch
from tqdm import tqdm

from torch_geometric.nn import GATConv, to_hetero


class HeteroGAT(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, num_layers, output_channels,
                 num_heads):
        super().__init__()
        self.model = to_hetero(
            GATForHetero(hidden_channels, num_layers, output_channels,
                         num_heads), metadata)  # TODO: replace by basic_gnn

    @torch.no_grad()
    def inference(self, loader, device, progress_bar=False):
        self.model.eval()
        if progress_bar:
            loader = tqdm(loader, desc="Inference")
        for batch in loader:
            batch = batch.to(device)
            self.model(batch.x_dict, batch.edge_index_dict)


class GATForHetero(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, out_channels, heads):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GATConv((-1, -1), hidden_channels, heads=heads,
                    add_self_loops=False))
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv((-1, -1), hidden_channels, heads=heads,
                        add_self_loops=False))
        self.convs.append(
            GATConv((-1, -1), out_channels, heads=heads, add_self_loops=False))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu_()
        return x
