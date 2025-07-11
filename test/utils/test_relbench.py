"""Tests for RelBench Integration Utilities.

Test suite for torch_geometric.utils.relbench module.

Author: Ahmed Gamal (aka AJamal27891)
Reference: GitHub Issue #9839 - Integrating GNNs and LLMs for Enhanced Data
Warehouse Understanding.
"""

import pytest
import torch

from torch_geometric.data import HeteroData

# Import with fallback for missing dependencies
try:
    from torch_geometric.utils.relbench import (
        RelBenchProcessor,
        create_relbench_hetero_data,
        get_warehouse_task_info,
    )
    RELBENCH_UTILS_AVAILABLE = True
except ImportError:
    RELBENCH_UTILS_AVAILABLE = False


@pytest.mark.skipif(not RELBENCH_UTILS_AVAILABLE,
                    reason="RelBench utilities not available")
class TestRelBenchProcessor:
    """Test suite for RelBenchProcessor class."""
    def test_processor_initialization(self):
        """Test RelBenchProcessor initialization."""
        processor = RelBenchProcessor()
        assert processor.sbert_model_name == "all-MiniLM-L6-v2"
        # Expected dimension for all-MiniLM-L6-v2
        assert processor.embedding_dim == 384

    def test_processor_custom_model(self):
        """Test RelBenchProcessor with custom SBERT model."""
        processor = RelBenchProcessor(sbert_model="all-mpnet-base-v2")
        assert processor.sbert_model_name == "all-mpnet-base-v2"
        # Expected dimension for all-mpnet-base-v2
        assert processor.embedding_dim == 768

    @pytest.mark.slow
    def test_process_dataset(self):
        """Test processing RelBench dataset to HeteroData."""
        processor = RelBenchProcessor()
        hetero_data = processor.process_dataset('rel-trial', sample_size=10)

        # Check HeteroData structure
        assert isinstance(hetero_data, HeteroData)
        assert len(hetero_data.node_types) > 0

        # Check node embeddings
        for node_type in hetero_data.node_types:
            if hasattr(hetero_data[node_type], 'x'):
                embeddings = hetero_data[node_type].x
                assert embeddings.shape[1] == 384  # SBERT embedding dimension
                assert embeddings.dtype == torch.float

    @pytest.mark.slow
    def test_warehouse_labels(self):
        """Test warehouse task label generation."""
        processor = RelBenchProcessor()
        hetero_data = processor.process_dataset('rel-trial', sample_size=5,
                                                add_warehouse_labels=True)

        # Check warehouse labels
        for node_type in hetero_data.node_types:
            if hasattr(hetero_data[node_type], 'x'):
                node_store = hetero_data[node_type]

                # Check lineage labels (0, 1, 2)
                assert hasattr(node_store, 'lineage_label')
                assert torch.all(node_store.lineage_label >= 0)
                assert torch.all(node_store.lineage_label < 3)

                # Check silo labels (0, 1)
                assert hasattr(node_store, 'silo_label')
                assert torch.all(node_store.silo_label >= 0)
                assert torch.all(node_store.silo_label < 2)

                # Check anomaly labels (0, 1)
                assert hasattr(node_store, 'anomaly_label')
                assert torch.all(node_store.anomaly_label >= 0)
                assert torch.all(node_store.anomaly_label < 2)

    def test_create_node_texts(self):
        """Test node text creation."""
        processor = RelBenchProcessor()

        # Mock table data
        import pandas as pd
        table_df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['test1', 'test2', 'test3'],
            'value': [10.5, 20.0, 30.5]
        })

        texts = processor._create_node_texts('test_table', table_df)

        assert len(texts) == 3
        assert all('Database table: test_table' in text for text in texts)
        assert all('id:' in text for text in texts)
        assert all('name:' in text for text in texts)
        assert all('value:' in text for text in texts)

    def test_generate_embeddings(self):
        """Test SBERT embedding generation."""
        processor = RelBenchProcessor()

        texts = [
            "Database table: test. id: 1. name: test1",
            "Database table: test. id: 2. name: test2"
        ]

        embeddings = processor._generate_embeddings(texts)

        assert embeddings.shape == (2, 384)
        assert embeddings.dtype == torch.float
        assert not torch.isnan(embeddings).any()


@pytest.mark.skipif(not RELBENCH_UTILS_AVAILABLE,
                    reason="RelBench utilities not available")
def test_create_relbench_hetero_data():
    """Test convenience function for creating HeteroData."""
    hetero_data = create_relbench_hetero_data('rel-trial', sample_size=5)

    assert isinstance(hetero_data, HeteroData)
    assert len(hetero_data.node_types) > 0


def test_get_warehouse_task_info():
    """Test warehouse task information retrieval."""
    task_info = get_warehouse_task_info()

    # Check structure
    assert 'lineage' in task_info
    assert 'silo' in task_info
    assert 'anomaly' in task_info

    # Check lineage task info
    lineage_info = task_info['lineage']
    assert lineage_info['num_classes'] == 3
    assert lineage_info['classes'] == ['source', 'intermediate', 'target']
    assert 'description' in lineage_info

    # Check silo task info
    silo_info = task_info['silo']
    assert silo_info['num_classes'] == 2
    assert silo_info['classes'] == ['connected', 'isolated']
    assert 'description' in silo_info

    # Check anomaly task info
    anomaly_info = task_info['anomaly']
    assert anomaly_info['num_classes'] == 2
    assert anomaly_info['classes'] == ['normal', 'anomaly']
    assert 'description' in anomaly_info


@pytest.mark.skipif(not RELBENCH_UTILS_AVAILABLE,
                    reason="RelBench utilities not available")
class TestEdgeCreation:
    """Test edge creation functionality."""
    def test_sample_edges(self):
        """Test sample edge creation."""
        processor = RelBenchProcessor()

        # Create minimal HeteroData
        hetero_data = HeteroData()
        hetero_data['studies'].num_nodes = 10
        hetero_data['conditions_studies'].num_nodes = 15

        # Add sample edges
        processor._add_sample_edges(hetero_data, 'studies', 'has_condition',
                                    'conditions_studies')

        # Check edges were created
        assert ('studies', 'has_condition',
                'conditions_studies') in hetero_data.edge_types
        assert ('conditions_studies', 'rev_has_condition',
                'studies') in hetero_data.edge_types

        # Check edge index shapes
        edge_index = hetero_data['studies', 'has_condition',
                                 'conditions_studies'].edge_index
        assert edge_index.shape[0] == 2  # Source and target nodes
        assert edge_index.shape[1] <= 20  # Maximum 20 edges

        # Check reverse edges
        rev_edge_index = hetero_data['conditions_studies', 'rev_has_condition',
                                     'studies'].edge_index
        assert torch.equal(edge_index, rev_edge_index.flip(0))


if __name__ == "__main__":
    pytest.main([__file__])
