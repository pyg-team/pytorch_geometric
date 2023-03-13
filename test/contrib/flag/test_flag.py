from collections import defaultdict
from typing import Any

import numpy as np
import pytest
import torch

from torch_geometric.contrib.flag import (
    FLAG,
    FLAGCallback,
    FLAGLossHistoryCallback,
    FLAGPerturbHistoryCallback,
)
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
    step_size = 0.5

    flag = FLAG(model, optimizer, loss_fn, device)
    train_idx = np.arange(3)

    expected_message = f"Invalid n_ascent_steps: {n_ascent_steps}." \
        + " n_ascent_steps should be a positive integer."
    with pytest.raises(ValueError, match=expected_message):
        flag(x, edge_index, y_true, train_idx, step_size, n_ascent_steps)


@pytest.mark.parametrize('step_size', [-10, 0, "pyg", None])
def test_flag_step_size_validation(model: torch.nn.Module,
                                   optimizer: torch.nn.Module,
                                   loss_fn: torch.nn.Module,
                                   device: torch.device, step_size: Any):
    x = torch.randn(3, 8)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    y_true = torch.tensor([1, 0, 0], dtype=torch.float).repeat(3, 1)

    flag = FLAG(model, optimizer, loss_fn, device)
    train_idx = np.arange(3)

    expected_message = f"Invalid step_size: {step_size}." \
        + " step_size should be a positive float."
    with pytest.raises(ValueError, match=expected_message):
        flag(x, edge_index, y_true, train_idx, step_size)


@pytest.mark.parametrize('num_nodes, num_labeled_nodes', [(3, 3), (10, 5),
                                                          (1000, 750)])
def test_output_size(model: torch.nn.Module, optimizer: torch.nn.Module,
                     loss_fn: torch.nn.Module, device: torch.device,
                     num_nodes: int, num_labeled_nodes: int):
    x = torch.randn(num_nodes, 8)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    y_true = torch.tensor([1, 0, 0], dtype=torch.float).repeat(num_nodes, 1)

    flag = FLAG(model, optimizer, loss_fn, device)
    train_idx = np.arange(num_labeled_nodes)
    step_size = 0.5

    flag = FLAG(model, optimizer, loss_fn, device)
    loss, out = flag(x, edge_index, y_true, train_idx, step_size)

    assert out.size() == (
        num_labeled_nodes,
        3), f"out should have size ({num_labeled_nodes}, 3). Got {out.size()}."
    assert loss.dtype == torch.float32, \
        f"loss should be a torch.float32 object. Got {loss.dtype}."


def test_zero_grad(model: torch.nn.Module, optimizer: torch.nn.Module,
                   loss_fn: torch.nn.Module, device: torch.device):
    x = torch.randn(3, 8)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    y_true = torch.tensor([1, 0, 0], dtype=torch.float).repeat(3, 1)
    step_size = 0.5
    train_idx = np.arange(3)

    flag = FLAG(model, optimizer, loss_fn, device)
    loss, out = flag(x, edge_index, y_true, train_idx, step_size)

    for group in optimizer.param_groups:
        for p in group['params']:
            assert torch.count_nonzero(
                p.grad) == 0, "After running FLAG, the gradients should be 0!"


class FLAGCounterCallback(FLAGCallback):
    def __init__(self):
        self.counts = defaultdict(int)

    def on_ascent_step_begin(self, step_index, loss):
        self.counts['on_ascent_step_begin'] += 1

    def on_ascent_step_end(self, step_index, loss, perturb_data):
        self.counts['on_ascent_step_end'] += 1

    def on_optimizer_step_begin(self, loss):
        self.counts['on_optimizer_step_begin'] += 1

    def on_optimizer_step_end(self, loss):
        self.counts['on_optimizer_step_end'] += 1


def test_flag_callbacks(model: torch.nn.Module, optimizer: torch.nn.Module,
                        loss_fn: torch.nn.Module, device: torch.device):

    loss_history_callback = FLAGLossHistoryCallback()
    perturb_history_callback = FLAGPerturbHistoryCallback()
    counter_callback = FLAGCounterCallback()

    number_of_ascent_steps = 10
    x = torch.randn(3, 8)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    y_true = torch.tensor([1, 0, 0], dtype=torch.float).repeat(3, 1)
    step_size = 0.5
    train_idx = np.arange(3)

    flag = FLAG(
        model, optimizer, loss_fn, device,
        [loss_history_callback, perturb_history_callback, counter_callback])
    loss, out = flag(x, edge_index, y_true, train_idx, step_size,
                     number_of_ascent_steps)

    assert counter_callback.counts[
        'on_ascent_step_begin'] == number_of_ascent_steps
    assert counter_callback.counts[
        'on_ascent_step_end'] == number_of_ascent_steps
    assert counter_callback.counts['on_optimizer_step_begin'] == 1
    assert counter_callback.counts['on_optimizer_step_end'] == 1

    # make sure the loss history is equal to the number of ascent steps,
    #   and that each loss value is unique
    loss_history = loss_history_callback.loss_history
    assert len(loss_history) == number_of_ascent_steps

    # likewise, for the perturb data history
    perturb_history = perturb_history_callback.perturb_history
    assert len(perturb_history) == number_of_ascent_steps
