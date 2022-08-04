from itertools import product

import pytest
import torch
from torch.autograd import gradcheck

from torch_geometric.utils import scatter
from torch_geometric.utils.torch_version import torch_version_minor

from .utils import devices, reductions, tensor

tests = [
    {
        'src': [1, 3, 2, 4, 5, 6],
        'index': [0, 1, 0, 1, 1, 3],
        'dim': -1,
        'sum': [3, 12, 0, 6],
        'mul': [2, 60, 0, 6],
        'mean': [1.5, 4, 0, 6],
        'min': [1, 3, 0, 6],
        'max': [2, 5, 0, 6],
    },
    {
        'src': [[1, 2], [5, 6], [3, 4], [7, 8], [9, 10], [11, 12]],
        'index': [0, 1, 0, 1, 1, 3],
        'dim': 0,
        'sum': [[4, 6], [21, 24], [0, 0], [11, 12]],
        'mul': [[1 * 3, 2 * 4], [5 * 7 * 9, 6 * 8 * 10], [0, 0], [11, 12]],
        'mean': [[2, 3], [7, 8], [0, 0], [11, 12]],
        'min': [[1, 2], [5, 6], [0, 0], [11, 12]],
        'max': [[3, 4], [9, 10], [0, 0], [11, 12]],
    },
    {
        'src': [[1, 5, 3, 7, 9, 11], [2, 4, 8, 6, 10, 12]],
        'index': [[0, 1, 0, 1, 1, 3], [0, 0, 1, 0, 1, 2]],
        'dim': 1,
        'sum': [[4, 21, 0, 11], [12, 18, 12, 0]],
        'mul': [[1 * 3, 5 * 7 * 9, 0, 11], [2 * 4 * 6, 8 * 10, 12, 0]],
        'mean': [[2, 7, 0, 11], [4, 9, 12, 0]],
        'min': [[1, 5, 0, 11], [2, 8, 12, 0]],
        'max': [[3, 9, 0, 11], [6, 10, 12, 0]],
    },
    {
        'src': [[[1, 2], [5, 6], [3, 4]], [[10, 11], [7, 9], [12, 13]]],
        'index': [[0, 1, 0], [2, 0, 2]],
        'dim': 1,
        'sum': [[[4, 6], [5, 6], [0, 0]], [[7, 9], [0, 0], [22, 24]]],
        'mul': [[[3, 8], [5, 6], [0, 0]], [[7, 9], [0, 0], [120, 11 * 13]]],
        'mean': [[[2, 3], [5, 6], [0, 0]], [[7, 9], [0, 0], [11, 12]]],
        'min': [[[1, 2], [5, 6], [0, 0]], [[7, 9], [0, 0], [10, 11]]],
        'max': [[[3, 4], [5, 6], [0, 0]], [[7, 9], [0, 0], [12, 13]]],
    },
    {
        'src': [[1, 3], [2, 4]],
        'index': [[0, 0], [0, 0]],
        'dim': 1,
        'sum': [[4], [6]],
        'mul': [[3], [8]],
        'mean': [[2], [3]],
        'min': [[1], [2]],
        'max': [[3], [4]],
    },
    {
        'src': [[[1, 1], [3, 3]], [[2, 2], [4, 4]]],
        'index': [[0, 0], [0, 0]],
        'dim': 1,
        'sum': [[[4, 4]], [[6, 6]]],
        'mul': [[[3, 3]], [[8, 8]]],
        'mean': [[[2, 2]], [[3, 3]]],
        'min': [[[1, 1]], [[2, 2]]],
        'max': [[[3, 3]], [[4, 4]]],
    },
]


@pytest.mark.skipif(torch_version_minor() < 12,
                    reason='requires torch=1.12.0 or higher')
@pytest.mark.parametrize('test,reduce,device',
                         product(tests, reductions, devices))
def test_forward(test, reduce, device):
    src = tensor(test['src'], device)
    index = tensor(test['index'], device, torch.long)
    dim = test['dim']
    expected = tensor(test[reduce], device)

    out = scatter(src, index, dim, reduce=reduce)

    assert torch.all(out == expected)


@pytest.mark.skipif(torch_version_minor() < 12,
                    reason='requires torch=1.12.0 or higher')
@pytest.mark.parametrize('test,reduce,device',
                         product(tests, reductions, devices))
def test_backward(test, reduce, device):
    src = tensor(test['src'], device, torch.double)
    src.requires_grad_()
    index = tensor(test['index'], device, torch.long)
    dim = test['dim']

    assert gradcheck(scatter, (src, index, dim, None, None, reduce))
