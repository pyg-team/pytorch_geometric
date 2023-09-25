import numpy as np
import torch
from torch import Tensor

from torch_geometric.sampler import SamplerOutput

from torch_geometric.distributed.utils import remove_duplicates


def test_remove_duplicates():
    node = np.array([0, 1, 2, 3])
    out_node = torch.tensor([0, 4, 1, 5, 1, 6, 2, 7, 3, 8])

    out = SamplerOutput(out_node, None, None, None)

    src, node, _, _ = remove_duplicates(out, node)

    expected_src = torch.tensor([4, 5, 6, 7, 8])
    expected_node = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])

    assert(torch.equal(src, expected_src))
    assert(np.array_equal(node, expected_node))


def test_remove_duplicates_disjoint():
    node = np.array([0, 1, 2, 3])
    batch = np.array([0, 1, 2, 3])

    out_node = torch.tensor([0, 4, 1, 5, 1, 6, 2, 6, 7, 3, 8])
    out_batch = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])

    out = SamplerOutput(out_node, None, None, None, out_batch)

    src, node, src_batch, batch = remove_duplicates(out, node, batch, disjoint=True)

    expected_src = torch.tensor([4, 5, 6, 7, 8])
    expected_node = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])

    expected_src_batch = torch.tensor([0, 1, 2, 3, 3])
    expected_batch = np.array([0, 1, 2, 3, 0, 1, 2, 3, 3])

    assert(torch.equal(src, expected_src))
    assert(np.array_equal(node, expected_node))

    assert(torch.equal(src_batch, expected_src_batch))
    assert(np.array_equal(batch, expected_batch))
