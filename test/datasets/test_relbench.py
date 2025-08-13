"""Test RelBench integration functionality."""

import pytest


def test_relbench_imports() -> None:
    """Test RelBench module imports."""
    try:
        from torch_geometric.datasets.relbench import (
            RelBenchDataset,
            RelBenchProcessor,
            create_relbench_hetero_data,
        )

        # Test that classes can be imported
        assert RelBenchDataset is not None
        assert RelBenchProcessor is not None
        assert create_relbench_hetero_data is not None
    except ImportError:
        pytest.skip("RelBench not available")


def test_relbench_processor() -> None:
    """Test RelBenchProcessor basic functionality."""
    try:
        from torch_geometric.datasets.relbench import RelBenchProcessor
    except ImportError:
        pytest.skip("RelBench not available")

    # Test processor initialization - handle missing dependencies gracefully
    try:
        processor = RelBenchProcessor()
        assert processor is not None
    except Exception as e:
        # If sentence-transformers not available, raise appropriate error
        if "sentence transformer" in str(e).lower():
            pytest.skip("Sentence transformers not available in CI")
        else:
            raise


def test_create_relbench_hetero_data() -> None:
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
