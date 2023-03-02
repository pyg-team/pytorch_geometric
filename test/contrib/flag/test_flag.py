from typing import Any

import numpy as np
import pytest
import torch

from torch_geometric.contrib.flag import FLAG
from torch_geometric.nn.models import GCN


@pytest.fixture
def model():
    gcn = GCN(8, 16, num_layers=3, out_channels=3)
    return gcn


@pytest.fixture
def optimizer(model):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    return optimizer


@pytest.fixture
def loss_fn():
    loss = torch.nn.CrossEntropyLoss()
    return loss


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.mark.parametrize('n_ascent_steps', [-10, 0, 42.8, "pyg", None])
def test_flag_n_ascent_steps_validation(model: torch.nn.Module,
                                        optimizer: torch.nn.Module,
                                        loss_fn: torch.nn.Module,
                                        device: torch.device,
                                        n_ascent_steps: Any):
    x = torch.randn(3, 8)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    y_true = torch.tensor([1, 0, 0], dtype=torch.float).repeat(3, 1)

    flag = FLAG(model, optimizer, loss_fn, device)
    train_idx = np.arange(3)

    step_size = 0.5

    expected_message = f"Invalid n_ascent_steps: {n_ascent_steps}." \
        + " n_ascent_steps should be a positive integer."
    with pytest.raises(ValueError, match=expected_message):
        flag(x, edge_index, y_true, train_idx, step_size, n_ascent_steps)
