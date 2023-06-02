import pytest
import torch

from torch_geometric.profile import benchmark
from torch_geometric.testing import withCUDA
from torch_geometric.utils.map import map_index


@withCUDA
@pytest.mark.parametrize('max_index', [3, 100_000_000])
def test_map_index(device, max_index):
    src = torch.tensor([2, 0, 1, 0, max_index], device=device)
    index = torch.tensor([max_index, 2, 0, 1], device=device)

    out, mask = map_index(src, index, inclusive=True)
    assert out.device == device
    assert mask is None
    assert out.tolist() == [1, 2, 3, 2, 0]


@withCUDA
@pytest.mark.parametrize('max_index', [3, 100_000_000])
def test_map_index_na(device, max_index):
    src = torch.tensor([2, 0, 1, 0, max_index], device=device)
    index = torch.tensor([max_index, 2, 0], device=device)

    out, mask = map_index(src, index, inclusive=False)
    assert out.device == device
    assert mask.device == device
    assert out.tolist() == [1, 2, 2, 0]
    assert mask.tolist() == [True, True, False, True, True]


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    src = torch.randint(0, 50_000_000, (100_000, ), device=args.device)
    index = src.unique()[:50_000]

    def trivial_map(src, index):
        mask = src.new_full((src.max() + 1, ), -1)
        mask[index] = torch.arange(index.numel(), device=index.device)

        return mask[src], None

    benchmark(
        funcs=[map_index, trivial_map],
        func_names=['map_index', 'trivial'],
        args=(src, index),
        num_steps=100 if args.device == 'cpu' else 100,
        num_warmups=50 if args.device == 'cpu' else 50,
    )
