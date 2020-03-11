import torch


class GNNExplainer(torch.nn.Module):
    def __init__(self, model, num_edges, num_nodes, in_channels):
        # Generate [num_nodes, in_channels] soft-mask.
        # Generate [num_edges] soft-mask for each MP layer in model.
        pass

    def forward(self, **kwargs):
        # Optimizes the masks.
        pass

    def optimize(self, **kwargs):
        pass

    def visualize(self, **kwargs):
        # Generates
        pass
