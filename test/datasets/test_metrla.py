import sys

from torch_geometric.datasets import MetrLa
import random
from pathlib import Path


def test_metrla():
    tag = 'metr_la'
    root = Path('/') / 'tmp' / tag
    n_previous_steps = 6
    n_future_steps = 6

    n_samples = 34272

    dataset = MetrLa(root=root,
                     n_previous_steps=n_previous_steps,
                     n_future_steps=n_future_steps)

    expected_dataset_len = 34260

    # Sanity checks
    assert len(dataset.gdrive_ids) == 4

    # Path assertions
    assert dataset.raw_dir == f'/tmp/{tag}/raw'
    assert dataset.processed_dir == f'/tmp/{tag}/processed'

    assert dataset.dataset_len == expected_dataset_len

    data = dataset[0]
    assert len(data) == 4

    assert data.x.size() == (6, 207, 1)
    assert data.y.size() == (6, 207, 1)