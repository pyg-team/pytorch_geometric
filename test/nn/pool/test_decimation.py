import torch

from torch_geometric.nn.pool.decimation import decimation_indices


def test_decimation_basic():
    # params
    N_1, N_2 = 4, 6
    decimation_factor = 2
    ptr = torch.LongTensor([0, N_1, N_1 + N_2])

    idx_decim, ptr_decim = decimation_indices(ptr, decimation_factor)
    size_decim = (N_1 // decimation_factor) + (N_2 // decimation_factor)
    assert idx_decim.size(0) == size_decim
    assert torch.equal(
        ptr_decim, torch.LongTensor([0, N_1 // decimation_factor, size_decim]))

    # usage
    # x = torch.randn(N_1 + N_2, 4)
    # x_decim = x[idx_decim]


def test_decimation_single_cloud():
    # params
    N_1 = 4
    decimation_factor = 2
    ptr = torch.LongTensor([0, N_1])
    idx_decim, ptr_decim = decimation_indices(ptr, decimation_factor)
    assert idx_decim.size(0) == N_1 // decimation_factor
    assert torch.equal(ptr_decim,
                       torch.LongTensor([0, N_1 // decimation_factor]))


def test_decimation_almost_empty():
    """Check that empty clouds are never created."""
    N_1 = 4
    decimation_factor = 666  # > than N_1
    ptr = torch.LongTensor([0, N_1])
    idx_decim, ptr_decim = decimation_indices(ptr, decimation_factor)
    assert idx_decim.size(0) == 1
    assert torch.equal(ptr_decim, torch.LongTensor([0, 1]))
