import torch
from tqdm import tqdm

from torch_geometric.nn import GAT, to_hetero


class HeteroGAT(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, num_layers, output_channels,
                 num_heads):
        super().__init__()
        self.model = to_hetero(
            GAT((-1, -1), hidden_channels, num_layers, output_channels,
                add_self_loops=False, heads=num_heads), metadata)

    def forward(self, x_dict, edge_index_dict):
        return self.model(x_dict, edge_index_dict)

    @torch.no_grad()
    def inference(self, loader, device, progress_bar=False, **kwargs):
        self.model.eval()
        if progress_bar:
            loader = tqdm(loader, desc="Inference")
        for batch in loader:
            batch = batch.to(device)
            if 'adj_t' in batch:
                self.model(batch.x_dict, batch.adj_t_dict)
            else:
                self.model(batch.x_dict, batch.edge_index_dict)

    @torch.no_grad()
    def test(self, x, loader, device, progress_bar=False):
        self.model.eval()
        total_examples = total_correct = 0
        if progress_bar:
            loader = tqdm(loader, desc="Evaluate")
        for batch in loader:
            batch = batch.to(device)
            if 'adj_t' in batch:
                out = self.model(batch.x_dict, batch.adj_t_dict)
            else:
                out = self.model(batch.x_dict, batch.edge_index_dict)
            batch_size = batch['paper'].batch_size
            out = out['paper'][:batch_size]
            pred = out.argmax(dim=-1)

            total_examples += batch_size
            total_correct += int((pred == batch['paper'].y[:batch_size]).sum())

        return total_correct / total_examples
