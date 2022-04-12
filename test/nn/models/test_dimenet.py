import pytest
import torch
import torch.nn as nn

from torch_geometric.data import Data
from torch_geometric.nn.models.dimenet import DimeNetPlusPlus


class TestDimeNetPlusPlus:
    def setup(self):
        n_atoms = 20
        z = torch.randint(1, 10, size=(n_atoms, ))
        pos = torch.randn(n_atoms, 3)
        self.data = Data(
            z=z,
            pos=pos)  # A molecule's atomic numbers and positional coordinates
        self.model = DimeNetPlusPlus(hidden_channels=5, out_channels=1,
                                     num_blocks=5, out_emb_channels=3,
                                     int_emb_size=5, basis_emb_size=5,
                                     num_spherical=5, num_radial=5,
                                     num_before_skip=2, num_after_skip=2)

    def test_output(self):
        with torch.no_grad():
            out = self.model(self.data.z, self.data.pos)
        assert out.shape[0] == 1

    def test_overfit(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=1, min_lr=0.00001)
        loss_fn = nn.L1Loss()
        y_target = torch.Tensor([1])
        for i in range(1000):
            optimizer.zero_grad()
            out = self.model(self.data.z, self.data.pos)
            loss = loss_fn(out, y_target)
            loss.backward()
            optimizer.step()

        assert loss.item() < 10
