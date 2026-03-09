import torch
from torch import Tensor

import torch_geometric.utils.langevin as langevin


def is_symmetric(adjs):
    return torch.allclose(adjs, adjs.transpose(-1, -2))


def mock_score_func(adjs: Tensor, node_flags: Tensor) -> Tensor:
    return adjs


def test_generate_initial_sample():
    torch.manual_seed(12345)

    batch_size = 4
    num_nodes = 3
    adjs, node_flags = langevin.generate_initial_sample(batch_size, num_nodes)

    assert adjs.shape == (batch_size, num_nodes, num_nodes)
    assert is_symmetric(adjs)
    assert torch.all(adjs.ge(0))

    assert node_flags.shape == (batch_size, num_nodes)
    assert torch.all((node_flags == 0) | (node_flags == 1))


def test_sample():
    torch.manual_seed(12345)
    batch_size = 4
    num_nodes = 3
    adjs = 0.5 + torch.randn((batch_size, num_nodes, num_nodes))
    node_flags = torch.ones((batch_size, num_nodes))

    new_adjs = langevin.sample(mock_score_func, adjs, node_flags, num_steps=10)
    assert new_adjs.shape == adjs.shape

    new_adjs = langevin.sample(mock_score_func, adjs, node_flags, num_steps=10,
                               quantize=True)
    assert new_adjs.shape == adjs.shape
    assert torch.all(~torch.eq(new_adjs, adjs))  # all values are different

    # check the quantization: all values are either 0 or 1
    num_zeros = (new_adjs == 0).sum()
    num_ones = (new_adjs == 1).sum()
    assert num_zeros > 0
    assert num_ones > 0
    assert num_zeros + num_ones == batch_size * num_nodes * num_nodes


def test_sample_step():
    torch.manual_seed(12345)

    batch_size = 4
    num_nodes = 3
    adjs = torch.ones((batch_size, num_nodes, num_nodes))
    node_flags = torch.ones((batch_size, num_nodes))

    new_adjs = langevin._sample_step(mock_score_func, adjs, node_flags,
                                     noise_variance=0.1, grad_step_size=0.5)

    assert new_adjs.shape == adjs.shape
    assert torch.all(~torch.eq(new_adjs, adjs))  # all values are different


def test_mask_adjs():
    batch_size = 4
    num_nodes = 3

    adjs = torch.ones((batch_size, num_nodes, num_nodes))
    node_flags = torch.ones((batch_size, num_nodes))

    masked = langevin.mask_adjs(adjs, node_flags)
    assert torch.all(masked == 1)

    node_flags = torch.zeros((batch_size, num_nodes))
    masked = langevin.mask_adjs(adjs, node_flags)
    assert torch.all(masked == 0)

    node_flags[0, :] = 1
    node_flags[1, :2] = 1
    node_flags[2, :1] = 1

    masked = langevin.mask_adjs(adjs, node_flags)
    # adjs[0] has 9 ones
    # adjs[1] has 4 ones
    # adjs[2] has 1 ones
    # adjs[3] has 0 ones
    assert masked.sum() == 14
