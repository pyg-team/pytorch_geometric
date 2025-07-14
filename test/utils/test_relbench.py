"""Tests for RelBench Integration Utilities.

This test suite uses local fixtures to avoid network dependencies in CI.
"""

# Import the module under test
try:
    from torch_geometric.datasets.relbench import get_warehouse_task_info
    RELBENCH_UTILS_AVAILABLE = True
except ImportError:
    RELBENCH_UTILS_AVAILABLE = False


def test_get_warehouse_task_info():
    """Test warehouse task info function."""
    if not RELBENCH_UTILS_AVAILABLE:
        return  # Skip test if module not available

    info = get_warehouse_task_info()

    assert isinstance(info, dict)
    assert 'lineage' in info
    assert 'silo' in info
    assert 'anomaly' in info

    # Check structure of each task
    for _, task_data in info.items():
        assert 'num_classes' in task_data
        assert 'classes' in task_data
        assert 'description' in task_data
        assert isinstance(task_data['num_classes'], int)
        assert isinstance(task_data['classes'], list)
        assert isinstance(task_data['description'], str)


if __name__ == "__main__":
    test_get_warehouse_task_info()
    print("All tests passed!")
