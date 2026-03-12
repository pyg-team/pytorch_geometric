import os
import pytest

from torch_geometric.data import Data

from torch_geometric.datasets import ECHOBenchmark  # noqa: E402


@pytest.mark.parametrize("task", ["sssp", "diam", "ecc", "energy", "charge"])
@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_echo_smoke(tmp_path, task, split):
    root = tmp_path / "ECHOBenchmark"
    ds = ECHOBenchmark(root=str(root), task=task, split=split)
    assert len(ds) >= 0
    if len(ds) > 0:
        assert isinstance(ds[0], Data)