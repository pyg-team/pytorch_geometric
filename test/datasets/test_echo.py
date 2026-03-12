import os
import pytest

from torch_geometric.data import Data

# Skip by default: PyG CI should not download large remote artifacts.
if os.getenv("PYG_TEST_WITH_NETWORK", "0") != "1":
    pytest.skip("Skipping ECHO download test (requires network).",
                allow_module_level=True)

from torch_geometric.datasets import ECHO  # noqa: E402


@pytest.mark.parametrize("task", ["sssp", "diam", "ecc", "energy", "charge"])
@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_echo_smoke(tmp_path, task, split):
    root = tmp_path / "ECHO"
    ds = ECHO(root=str(root), task=task, split=split)
    assert len(ds) >= 0
    if len(ds) > 0:
        assert isinstance(ds[0], Data)