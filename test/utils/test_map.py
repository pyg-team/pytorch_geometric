import pytest
import torch

from torch_geometric.profile import benchmark
from torch_geometric.testing import withCUDA, withPackage
from torch_geometric.utils.map import map_index


@withCUDA
@withPackage('pandas')
@pytest.mark.parametrize('max_index', [3, 100_000_000])
def test_map_index(device, max_index):
    src = torch.tensor([2, 0, 1, 0, max_index], device=device)
    index = torch.tensor([max_index, 2, 0, 1], device=device)

    out, mask = map_index(src, index, inclusive=True)
    assert out.device == device
    assert mask is None
    assert out.tolist() == [1, 2, 3, 2, 0]


@withCUDA
@withPackage('pandas')
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

    src = torch.randint(0, 100_000_000, (100_000, ), device=args.device)
    index = src.unique()

    def trivial_map(src, index, max_index, inclusive):
        if max_index is None:
            max_index = max(src.max(), index.max())

        if inclusive:
            assoc = src.new_empty(max_index + 1)
        else:
            assoc = src.new_full((max_index + 1, ), -1)
        assoc[index] = torch.arange(index.numel(), device=index.device)
        out = assoc[src]

        if inclusive:
            return out, None
        else:
            mask = out != -1
            return out[mask], mask

    print('Inclusive:')
    benchmark(
        funcs=[trivial_map, map_index],
        func_names=['trivial', 'map_index'],
        args=(src, index, None, True),
        num_steps=100,
        num_warmups=50,
    )

    print('Exclusive:')
    benchmark(
        funcs=[trivial_map, map_index],
        func_names=['trivial', 'map_index'],
        args=(src, index[:50_000], None, False),
        num_steps=100,
        num_warmups=50,
    )
