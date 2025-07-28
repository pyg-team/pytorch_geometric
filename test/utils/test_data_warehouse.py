"""Comprehensive tests for data warehouse intelligence system."""

import pytest
import torch


class TestWarehouseTaskHead:
    """Test multi-task head functionality."""
    def test_task_head_initialization(self):
        """Test WarehouseTaskHead initialization."""
        try:
            from torch_geometric.utils.data_warehouse import WarehouseTaskHead
        except ImportError:
            pytest.skip("Data warehouse not available")

        task_head = WarehouseTaskHead(hidden_dim=256, num_lineage_types=5)
        assert task_head is not None
        assert hasattr(task_head, 'heads')
        assert 'lineage' in task_head.heads
        assert 'silo' in task_head.heads
        assert 'quality' in task_head.heads
        assert 'impact' in task_head.heads

    def test_all_task_predictions(self):
        """Test predictions for all warehouse tasks."""
        try:
            from torch_geometric.utils.data_warehouse import WarehouseTaskHead
        except ImportError:
            pytest.skip("Data warehouse not available")

        task_head = WarehouseTaskHead(hidden_dim=256)
        x = torch.randn(10, 256)

        # Test all task types
        tasks = ['lineage', 'silo', 'quality', 'impact']
        expected_shapes = [(10, 5), (10, 1), (10, 1), (10, 3)]

        for task, expected_shape in zip(tasks, expected_shapes):
            pred = task_head(x, task=task)
            assert pred.shape == expected_shape

    def test_invalid_task_handling(self):
        """Test handling of invalid task types."""
        try:
            from torch_geometric.utils.data_warehouse import WarehouseTaskHead
        except ImportError:
            pytest.skip("Data warehouse not available")

        task_head = WarehouseTaskHead()
        x = torch.randn(5, 2048)

        with pytest.raises(ValueError):
            task_head(x, task='invalid_task')


class TestWarehouseGRetriever:
    """Test G-Retriever integration."""
    def test_gretriever_initialization(self):
        """Test WarehouseGRetriever initialization."""
        try:
            from torch_geometric.utils.data_warehouse import \
                WarehouseGRetriever  # noqa: E501
        except ImportError:
            pytest.skip("Data warehouse not available")

        model = WarehouseGRetriever()
        assert model is not None
        assert hasattr(model, 'gnn')
        assert hasattr(model, 'llm')
        assert hasattr(model, 'g_retriever')
        assert hasattr(model, 'task_head')

    def test_inference_functionality(self):
        """Test LLM inference functionality."""
        try:
            from torch_geometric.utils.data_warehouse import \
                WarehouseGRetriever  # noqa: E501
        except ImportError:
            pytest.skip("Data warehouse not available")

        model = WarehouseGRetriever()
        x = torch.randn(20, 384)
        edge_index = torch.randint(0, 20, (2, 30))
        batch = torch.zeros(20, dtype=torch.long)

        result = model.inference(question=['What is the data lineage?'], x=x,
                                 edge_index=edge_index, batch=batch,
                                 max_out_tokens=50)

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], str)

    def test_task_prediction_functionality(self):
        """Test task-specific predictions."""
        try:
            from torch_geometric.utils.data_warehouse import \
                WarehouseGRetriever  # noqa: E501
        except ImportError:
            pytest.skip("Data warehouse not available")

        model = WarehouseGRetriever()
        x = torch.randn(15, 384)
        edge_index = torch.randint(0, 15, (2, 25))

        # Test all task types
        for task in ['lineage', 'silo', 'quality', 'impact']:
            pred = model.predict_task(x, edge_index, task=task)
            assert pred is not None
            assert pred.dim() >= 1


class TestWarehouseConversationSystem:
    """Test conversation system functionality."""
    def test_conversation_system_initialization(self):
        """Test WarehouseConversationSystem initialization."""
        try:
            from torch_geometric.utils.data_warehouse import (  # noqa: E501
                SimpleWarehouseModel,
                WarehouseConversationSystem,
            )
        except ImportError:
            pytest.skip("Data warehouse not available")

        model = SimpleWarehouseModel()
        conversation_system = WarehouseConversationSystem(model)

        assert conversation_system is not None
        assert hasattr(conversation_system, 'model')
        assert hasattr(conversation_system, 'conversation_history')
        assert len(conversation_system.conversation_history) == 0

    def test_query_processing_all_types(self):
        """Test processing of all query types."""
        try:
            from torch_geometric.utils.data_warehouse import (  # noqa: E501
                SimpleWarehouseModel,
                WarehouseConversationSystem,
            )
        except ImportError:
            pytest.skip("Data warehouse not available")

        model = SimpleWarehouseModel()
        conversation_system = WarehouseConversationSystem(model)

        x = torch.randn(25, 384)
        edge_index = torch.randint(0, 25, (2, 40))
        graph_data = {'x': x, 'edge_index': edge_index, 'batch': None}

        queries = [('What is the data lineage?', 'lineage'),
                   ('Are there data silos?', 'silo'),
                   ('How is data quality?', 'quality'),
                   ('Analyze impact', 'impact')]

        for query, expected_type in queries:
            result = conversation_system.process_query(query, graph_data)

            assert 'answer' in result
            assert 'query_type' in result
            assert 'predictions' in result
            assert result['query_type'] == expected_type

    def test_conversation_history_tracking(self):
        """Test conversation history functionality."""
        try:
            from torch_geometric.utils.data_warehouse import (  # noqa: E501
                SimpleWarehouseModel,
                WarehouseConversationSystem,
            )
        except ImportError:
            pytest.skip("Data warehouse not available")

        model = SimpleWarehouseModel()
        conversation_system = WarehouseConversationSystem(model)

        x = torch.randn(10, 384)
        edge_index = torch.randint(0, 10, (2, 15))
        graph_data = {'x': x, 'edge_index': edge_index, 'batch': None}

        # Process multiple queries
        queries = ['What is lineage?', 'Check data quality']
        for query in queries:
            conversation_system.process_query(query, graph_data)

        # Verify history
        assert len(conversation_system.conversation_history) == len(queries)

        for i, entry in enumerate(conversation_system.conversation_history):
            assert 'query' in entry
            assert 'answer' in entry
            assert 'timestamp' in entry
            assert entry['query'] == queries[i]


class TestSimpleWarehouseModel:
    """Test simplified warehouse model."""
    def test_simple_model_initialization(self):
        """Test SimpleWarehouseModel initialization."""
        try:
            from torch_geometric.utils.data_warehouse import \
                SimpleWarehouseModel  # noqa: E501
        except ImportError:
            pytest.skip("Data warehouse not available")

        model = SimpleWarehouseModel()
        assert model is not None
        assert hasattr(model, 'gnn')
        assert hasattr(model, 'task_head')

    def test_simple_model_forward_pass(self):
        """Test forward pass functionality."""
        try:
            from torch_geometric.utils.data_warehouse import \
                SimpleWarehouseModel  # noqa: E501
        except ImportError:
            pytest.skip("Data warehouse not available")

        model = SimpleWarehouseModel()
        x = torch.randn(20, 384)
        edge_index = torch.randint(0, 20, (2, 30))

        result = model(question=['What is the lineage?'], x=x,
                       edge_index=edge_index, task='lineage')
        assert result is not None
        assert 'pred' in result

    def test_simple_model_predict_task(self):
        """Test predict_task method."""
        try:
            from torch_geometric.utils.data_warehouse import \
                SimpleWarehouseModel  # noqa: E501
        except ImportError:
            pytest.skip("Data warehouse not available")

        model = SimpleWarehouseModel()
        x = torch.randn(15, 384)
        edge_index = torch.randint(0, 15, (2, 25))

        pred = model.predict_task(x, edge_index, task='silo')
        assert pred is not None
        assert pred.dim() >= 1


class TestWarehouseDemo:
    """Test warehouse demo functionality."""
    def test_create_warehouse_demo(self):
        """Test create_warehouse_demo function."""
        try:
            from torch_geometric.utils.data_warehouse import \
                create_warehouse_demo  # noqa: E501
        except ImportError:
            pytest.skip("Data warehouse not available")

        demo_system = create_warehouse_demo()
        assert demo_system is not None
        assert hasattr(demo_system, 'process_query')
        assert hasattr(demo_system, 'conversation_history')
        assert hasattr(demo_system, 'model')
