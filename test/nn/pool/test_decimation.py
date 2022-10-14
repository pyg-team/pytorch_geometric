import torch
from torch_geometric.nn.pool.decimation import decimation_indices


def test_decimation():
    # params
    N_1, N_2 = 4, 6
    decimation_factor = 2

    # data
    x = torch.randn(N_1 + N_2, 4)
    batch = torch.tensor([0 for _ in range(N_1)] + [1 for _ in range(N_2)])
    ptr = torch.LongTensor([0, N_1])

    idx_decim, ptr_decim = decimation_indices(ptr, decimation_factor)
    assert idx_decim.size(0) == (
        (N_1 // decimation_factor) + (N_2 // decimation_factor)
    )
    assert torch.equal(
        ptr_decim, torch.LongTensor([0, N_1 // decimation_factor])
    )

    # usage
    batch_decim = batch[idx_decim]
    x_decim = x[idx_decim]
