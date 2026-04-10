import os

import pytest

from torch_geometric.data import Data
from torch_geometric.datasets import ECHOBenchmark  # noqa: E402


@pytest.mark.parametrize("task", ["sssp", "diam", "ecc", "energy", "charge"])
@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_echo_smoke(tmp_path, task, split):
    root = os.path.join(tmp_path, "ECHOBenchmark")
    data = ECHOBenchmark(root=root, task=task, split=split)
    assert len(data) > 0
    assert isinstance(data[0], Data)
    assert data.num_features > 0
    assert data.num_edge_features >= 0
    assert data.num_classes == 1

    NODE_LVL_TASKS = ['sssp', 'ecc', 'charge']
    GRAPH_LVL_TASKS = ['diam', 'energy']

    assert data.is_node_level_task == (data.task in NODE_LVL_TASKS)
    assert not data.is_node_level_task == (data.task in GRAPH_LVL_TASKS)
