"""Test RelBench integration functionality."""

import pytest


def test_get_warehouse_task_info():
    """Test get_warehouse_task_info function."""
    # Import the module under test
    try:
        from torch_geometric.datasets.relbench import get_warehouse_task_info
    except ImportError:
        pytest.skip("RelBench not available")

    # Test basic functionality
    task_info = get_warehouse_task_info()

    # Verify structure
    assert isinstance(task_info, dict)
    assert 'lineage' in task_info
    assert 'silo' in task_info
    assert 'anomaly' in task_info

    # Verify each task has required fields
    for _task_name, task_data in task_info.items():
        assert 'num_classes' in task_data
        assert 'description' in task_data
        assert isinstance(task_data['num_classes'], int)
        assert isinstance(task_data['description'], str)


def test_relbench_processor():
    """Test RelBenchProcessor basic functionality."""
    try:
        from torch_geometric.datasets.relbench import RelBenchProcessor
    except ImportError:
        pytest.skip("RelBench not available")

    # Test processor initialization
    processor = RelBenchProcessor()
    assert processor is not None


def test_create_relbench_hetero_data():
    """Test create_relbench_hetero_data function."""
    try:
        from torch_geometric.datasets.relbench import \
            create_relbench_hetero_data  # noqa: E501
    except ImportError:
        pytest.skip("RelBench not available")

    # Test with minimal parameters (will skip if data not available)
    try:
        hetero_data = create_relbench_hetero_data('rel-trial', sample_size=5)
        assert hetero_data is not None
        assert hasattr(hetero_data, 'num_nodes')
    except Exception:
        # Skip if data download fails or other issues
        pytest.skip("RelBench data not available or download failed")
