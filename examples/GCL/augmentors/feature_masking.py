import torch
from GCL.augmentors.augmentor import Graph, Augmentor

class FeatureMasking(Augmentor):
    def __init__(self, pf: float):
        super(FeatureMasking, self).__init__()
        self.pf = pf

    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unfold()

        device = x.device
        drop_mask = torch.empty((x.size(1),), dtype=torch.float32).uniform_(0, 1) < self.pf
        drop_mask = drop_mask.to(device)
        x = x.clone()
        x[:, drop_mask] = 0

        return Graph(x=x, adj_t=edge_index, edge_weights=edge_weights)
