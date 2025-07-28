"""Comprehensive tests for RelBench dataset integration."""

import pytest
import torch

from torch_geometric.data import HeteroData


class TestRelBenchCore:
    """Test core RelBench functionality."""
    def test_create_relbench_hetero_data(self):
        """Test RelBench data creation with real database."""
        try:
            from torch_geometric.datasets.relbench import \
                create_relbench_hetero_data  # noqa: E501
        except ImportError:
            pytest.skip("RelBench not available")

        hetero_data = create_relbench_hetero_data('rel-f1', sample_size=10)

        assert isinstance(hetero_data, HeteroData)
        assert len(hetero_data.node_types) > 0
        assert len(hetero_data.edge_types) > 0

        # Verify node features
        for node_type in hetero_data.node_types:
            assert hetero_data[node_type].x is not None
            assert hetero_data[node_type].x.dim() == 2
            assert hetero_data[node_type].x.size(
                1) == 384  # SBERT embedding dim

        # Verify edge structure
        for edge_type in hetero_data.edge_types:
            edge_index = hetero_data[edge_type].edge_index
            assert edge_index is not None
            assert edge_index.dim() == 2
            assert edge_index.size(0) == 2

    def test_relbench_dataset_class(self):
        """Test RelBenchDataset wrapper class."""
        try:
            from torch_geometric.datasets.relbench import RelBenchDataset
        except ImportError:
            pytest.skip("RelBench not available")

        dataset = RelBenchDataset(root='/tmp/test_relbench',
                                  dataset_name='rel-f1', sample_size=5)

        assert len(dataset) > 0
        data = dataset[0]
        # Data is HeteroData, so check for node_types instead of x
        assert hasattr(data, 'node_types')
        assert len(data.node_types) > 0


class TestHeuristicLabeler:
    """Test heuristic labeling system."""
    def test_labeler_initialization(self):
        """Test HeuristicLabeler initialization."""
        try:
            from torch_geometric.datasets.relbench import HeuristicLabeler
        except ImportError:
            pytest.skip("RelBench not available")

        labeler = HeuristicLabeler()
        assert labeler is not None

    def test_generate_labels_all_tasks(self):
        """Test label generation for all warehouse tasks."""
        try:
            from torch_geometric.datasets.relbench import HeuristicLabeler
        except ImportError:
            pytest.skip("RelBench not available")

        hetero_data = HeteroData()
        hetero_data['entity'].x = torch.randn(10, 384)
        hetero_data['entity', 'relates',
                    'entity'].edge_index = torch.randint(0, 10, (2, 15))

        labeler = HeuristicLabeler()
        labels = labeler.generate_labels(
            hetero_data, None,
            tasks=['lineage', 'silo', 'anomaly', 'quality', 'impact'])

        assert isinstance(labels, dict)
        assert len(labels) >= 3  # At least lineage, silo, anomaly

        # Verify label dimensions (shapes may vary based on implementation)
        for _task, label_tensor in labels.items():
            assert label_tensor.shape[0] == 10  # Should match number of nodes
            assert label_tensor.dim() >= 1  # At least 1D tensor

    def test_foreign_key_analysis(self):
        """Test foreign key depth analysis."""
        try:
            from torch_geometric.datasets.relbench import HeuristicLabeler
        except ImportError:
            pytest.skip("RelBench not available")

        labeler = HeuristicLabeler()

        # Test with None database (should handle gracefully)
        depth_analysis = labeler._analyze_foreign_key_depth(None)
        assert isinstance(depth_analysis, dict)
        assert len(depth_analysis) == 0


class TestRelBenchProcessor:
    """Test RelBench data processing."""
    def test_processor_initialization(self):
        """Test RelBenchProcessor initialization."""
        try:
            from torch_geometric.datasets.relbench import RelBenchProcessor
        except ImportError:
            pytest.skip("RelBench not available")

        processor = RelBenchProcessor()
        assert processor is not None

    def test_relbench_data_processing(self):
        """Test RelBench data processing."""
        try:
            from torch_geometric.datasets.relbench import RelBenchProcessor
        except ImportError:
            pytest.skip("RelBench not available")

        processor = RelBenchProcessor()

        # Test text encoding functionality
        texts = ["sample text 1", "sample text 2"]
        embeddings = processor.encode_texts(texts)

        assert embeddings is not None
        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] == 384  # EMBEDDING_DIM

    def test_warehouse_labels_integration(self):
        """Test warehouse label integration."""
        try:
            from torch_geometric.datasets.relbench import \
                _add_warehouse_labels_to_hetero_data  # noqa: E501
        except ImportError:
            pytest.skip("RelBench not available")

        hetero_data = HeteroData()
        hetero_data['entity'].x = torch.randn(5, 384)
        hetero_data['entity', 'relates',
                    'entity'].edge_index = torch.randint(0, 5, (2, 8))

        labeled_data = _add_warehouse_labels_to_hetero_data(
            hetero_data, db=None)

        # Function returns the hetero_data with labels added
        assert labeled_data is not None
        assert isinstance(labeled_data, HeteroData)


class TestRelBenchError:
    """Test custom exception handling."""
    def test_relbench_error(self):
        """Test RelBenchError exception."""
        try:
            from torch_geometric.datasets.relbench import RelBenchError
        except ImportError:
            pytest.skip("RelBench not available")

        with pytest.raises(RelBenchError):
            raise RelBenchError("Test error message")
