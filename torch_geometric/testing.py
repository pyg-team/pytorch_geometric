import os
from typing import Callable


def is_full_test() -> bool:
    r"""Whether to run the full but time-consuming test suite."""
    return os.getenv('FULL_TEST', '0') == '1'


def onlyFullTest(func: Callable) -> Callable:
    r"""A decorator to specify that this function belongs to the full test
    suite."""

    import pytest
    return pytest.mark.skipif(not is_full_test(), reason="Fast test run")(func)


def withDataset(name: str, *args, **kwargs) -> Callable:
    r"""A decorator to load (and re-use) a variety of datasets during
    testing."""
    def decorator(func: Callable) -> Callable:
        if name.lower() in ['cora', 'citeseer', 'pubmed']:
            from torch_geometric.datasets import Planetoid
            dataset = Planetoid('/tmp/Planetoid', name, *args, **kwargs)
        elif name in ['BZR', 'ENZYMES', 'IMDB-BINARY', 'MUTAG']:
            from torch_geometric.datasets import TUDataset
            dataset = TUDataset('/tmp/TU', name, *args, **kwargs)
        else:
            print("FAIL", name)
            raise NotImplementedError

        import pytest
        return pytest.mark.parametrize('dataset', [dataset])(func)

    return decorator
