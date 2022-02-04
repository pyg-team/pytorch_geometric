import os.path as osp

import torch
import torch.nn.functional as F
from pytorch_lightning.lite import LightningLite

from torch_geometric.datasets import Flickr
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GAT

path = osp.join(osp.dirname(osp.realpath(__file__)), '../../data/Flickr')


class Lite(LightningLite):
    def run(self, path: str, num_workers: int = 0):
        dataset = Flickr(path)
        train_loader = NeighborLoader(dataset[0], num_neighbors=[10] * 2,
                                      shuffle=True, batch_size=1024,
                                      input_nodes=dataset[0].train_mask,
                                      num_workers=num_workers,
                                      persistent_workers=num_workers > 0)
        train_loader = self.setup_dataloaders(train_loader)

        model = GAT(dataset.num_features, hidden_channels=256, num_layers=2,
                    out_channels=dataset.num_classes, dropout=0.3, jk='cat')
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        model, optimizer = self.setup(model, optimizer)

        for epoch in range(1, 101):
            loss = self.train(train_loader, model, optimizer)
            print(loss)

    def train(self, model, loader, optimizer):
        model.train()
        total_loss = total_examples = 0
        for data in loader:
            optimizer.zero_grad()
            batch_size = data.batch_size
            out = model(data.x, data.edge_index)
            loss = F.cross_entropy(out[:batch_size], data.y[:batch_size])
            self.backward(loss)
            optimizer.step()
            total_loss += float(loss) * batch_size
            total_examples += batch_size
        return total_loss / total_examples


Lite(devices='auto', accelerator='auto').run(path, num_workers=6)
