import torch

from torch_geometric.nn.pool.decimation import decimation_indices


def test_decimation_basic():
    N_1, N_2 = 4, 6
    decimation_factor = 2
    ptr = torch.tensor([0, N_1, N_1 + N_2])

    idx_decim, ptr_decim = decimation_indices(ptr, decimation_factor)
    print(idx_decim, ptr_decim)

    expected_size = (N_1 // decimation_factor) + (N_2 // decimation_factor)
    assert idx_decim.size(0) == expected_size

    expected = torch.tensor([0, N_1 // decimation_factor, expected_size])
    assert torch.equal(ptr_decim, expected)


def test_decimation_single_cloud():
    N_1 = 4
    decimation_factor = 2
    ptr = torch.tensor([0, N_1])

    idx_decim, ptr_decim = decimation_indices(ptr, decimation_factor)

    expected_size = N_1 // decimation_factor
    assert idx_decim.size(0) == expected_size
    assert torch.equal(ptr_decim, torch.tensor([0, expected_size]))


def test_decimation_almost_empty():
    N_1 = 4
    decimation_factor = 666  # greater than N_1
    ptr = torch.tensor([0, N_1])

    idx_decim, ptr_decim = decimation_indices(ptr, decimation_factor)

    assert idx_decim.size(0) == 1
    assert torch.equal(ptr_decim, torch.tensor([0, 1]))
