import numpy as np
import torch

from torch_geometric.utils import lexsort


def test_lexsort():
    keys = [torch.randn(100) for _ in range(3)]

    expected = np.lexsort([key.numpy() for key in keys])
    assert torch.equal(lexsort(keys), torch.from_numpy(expected))
