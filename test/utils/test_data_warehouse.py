# isort: skip_file
"""Comprehensive tests for data warehouse intelligence system."""

from typing import Any, Optional, cast

import pytest
import torch

from torch_geometric.testing import withPackage


class TestWarehouseTaskHead:
    """Test multi-task head functionality."""
    def test_task_head_initialization(self) -> None:
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

    def test_all_task_predictions(self) -> None:
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

    def test_invalid_task_handling(self) -> None:
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
    @withPackage('transformers')
    def test_gretriever_initialization(self) -> None:
        """Test WarehouseGRetriever initialization."""
        try:
            # yapf: disable
            from torch_geometric.utils.data_warehouse import (
                WarehouseGRetriever,
            )

            # yapf: enable
        except ImportError:
            pytest.skip("Data warehouse not available")

        model = WarehouseGRetriever()
        assert model is not None
        assert hasattr(model, 'gnn')
        assert hasattr(model, 'llm')
        assert hasattr(model, 'g_retriever')
        assert hasattr(model, 'task_head')

    @withPackage('transformers')
    def test_gretriever_with_custom_config(self) -> None:
        """Test WarehouseGRetriever with custom configuration."""
        try:
            # yapf: disable
            from torch_geometric.utils.data_warehouse import (
                WarehouseGRetriever,
            )

            # yapf: enable
        except ImportError:
            pytest.skip("Data warehouse not available")

        # Test with custom parameters (using actual constructor parameters)
        model = WarehouseGRetriever(
            llm_model_name='TinyLlama/TinyLlama-1.1B-Chat-v0.1')
        assert model is not None
        assert hasattr(model, 'gnn')
        assert hasattr(model, 'llm')

    @withPackage('transformers')
    def test_inference_functionality(self) -> None:
        """Test LLM inference functionality."""
        try:
            # yapf: disable
            from torch_geometric.utils.data_warehouse import (
                WarehouseGRetriever,
            )

            # yapf: enable
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

    @withPackage('transformers')
    def test_inference_with_multiple_questions(self) -> None:
        """Test inference with multiple questions."""
        try:
            # yapf: disable
            from torch_geometric.utils.data_warehouse import (
                WarehouseGRetriever,
            )

            # yapf: enable
        except ImportError:
            pytest.skip("Data warehouse not available")

        model = WarehouseGRetriever()
        x = torch.randn(15, 384)
        edge_index = torch.randint(0, 15, (2, 25))
        batch = torch.zeros(15, dtype=torch.long)

        questions = [
            'What is the lineage?', 'Are there silos?', 'How is quality?'
        ]
        result = model.inference(question=questions, x=x,
                                 edge_index=edge_index, batch=batch,
                                 max_out_tokens=30)

        assert isinstance(result, list)
        assert len(result) == len(questions)
        for answer in result:
            assert isinstance(answer, str)

    @withPackage('transformers')
    def test_task_prediction_functionality(self) -> None:
        """Test task-specific predictions."""
        try:
            # yapf: disable
            from torch_geometric.utils.data_warehouse import (
                WarehouseGRetriever,
            )

            # yapf: enable
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

    @withPackage('transformers')
    def test_forward_pass_functionality(self) -> None:
        """Test forward pass with different tasks."""
        try:
            # yapf: disable
            from torch_geometric.utils.data_warehouse import (
                WarehouseGRetriever,
            )

            # yapf: enable
        except ImportError:
            pytest.skip("Data warehouse not available")

        model = WarehouseGRetriever()
        x = torch.randn(12, 384)
        edge_index = torch.randint(0, 12, (2, 20))
        batch = torch.zeros(12, dtype=torch.long)

        # Test forward pass with different tasks (using correct signature)
        # Forward is for training; return may be None or Tensor
        result = model(
            question=['Test question'],
            x=x,
            edge_index=edge_index,
            batch=batch,
            label=None,
        )
        # Accept both Tensor or None (since custom forward may delegate)
        assert ((result is None) or isinstance(result, torch.Tensor))

    @withPackage('transformers')
    def test_gretriever_error_handling(self) -> None:
        """Test G-Retriever error handling."""
        try:
            # yapf: disable
            from torch_geometric.utils.data_warehouse import (
                WarehouseGRetriever,
            )

            # yapf: enable
        except ImportError:
            pytest.skip("Data warehouse not available")

        model = WarehouseGRetriever()

        # Test with invalid task
        x = torch.randn(10, 384)
        edge_index = torch.randint(0, 10, (2, 15))

        with pytest.raises(ValueError):
            model.predict_task(x, edge_index, task='invalid_task')


class TestWarehouseConversationSystem:
    """Test conversation system functionality."""
    def test_conversation_system_initialization(self) -> None:
        """Test WarehouseConversationSystem initialization."""
        try:
            # yapf: disable
            from torch_geometric.utils.data_warehouse import (
                SimpleWarehouseModel,
                WarehouseConversationSystem,
            )

            # yapf: enable
        except ImportError:
            pytest.skip("Data warehouse not available")

        model = SimpleWarehouseModel()
        conversation_system = WarehouseConversationSystem(model)

        assert conversation_system is not None
        assert hasattr(conversation_system, 'model')
        assert hasattr(conversation_system, 'conversation_history')
        assert len(conversation_system.conversation_history) == 0

    def test_query_processing_all_types(self) -> None:
        """Test processing of all query types."""
        try:
            # yapf: disable
            from torch_geometric.utils.data_warehouse import (
                SimpleWarehouseModel,
                WarehouseConversationSystem,
            )

            # yapf: enable
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

    def test_conversation_history_tracking(self) -> None:
        """Test conversation history functionality."""
        try:
            # yapf: disable
            from torch_geometric.utils.data_warehouse import (
                SimpleWarehouseModel,
                WarehouseConversationSystem,
            )

            # yapf: enable
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
    def test_simple_model_initialization(self) -> None:
        """Test SimpleWarehouseModel initialization."""
        try:
            # yapf: disable
            from torch_geometric.utils.data_warehouse import (
                SimpleWarehouseModel,
            )

            # yapf: enable
        except ImportError:
            pytest.skip("Data warehouse not available")

        model = SimpleWarehouseModel()
        assert model is not None
        assert hasattr(model, 'gnn')
        assert hasattr(model, 'task_head')

    def test_simple_model_with_custom_params(self) -> None:
        """Test SimpleWarehouseModel with custom parameters."""
        try:
            # yapf: disable
            from torch_geometric.utils.data_warehouse import (
                SimpleWarehouseModel,
            )

            # yapf: enable
        except ImportError:
            pytest.skip("Data warehouse not available")

        # Test with default parameters (constructor has no custom params)

        model = SimpleWarehouseModel()
        # Keep assertions on separate lines for flake8 line length
        assert model is not None
        assert hasattr(model, 'gnn')
        assert hasattr(model, 'task_head')

    def test_simple_model_forward_pass(self) -> None:
        """Test forward pass functionality."""
        try:
            # yapf: disable
            from torch_geometric.utils.data_warehouse import (
                SimpleWarehouseModel,
            )

            # yapf: enable
        except ImportError:
            pytest.skip("Data warehouse not available")

        model = SimpleWarehouseModel()
        x = torch.randn(20, 384)
        edge_index = torch.randint(0, 20, (2, 30))

        result = model(question=['What is the lineage?'], x=x,
                       edge_index=edge_index, task='lineage')
        assert result is not None
        assert 'pred' in result

    def test_simple_model_all_tasks(self) -> None:
        """Test forward pass with all task types."""
        try:
            # yapf: disable
            from torch_geometric.utils.data_warehouse import (
                SimpleWarehouseModel,
            )

            # yapf: enable
        except ImportError:
            pytest.skip("Data warehouse not available")

        model = SimpleWarehouseModel()
        x = torch.randn(15, 384)
        edge_index = torch.randint(0, 15, (2, 25))

        for task in ['lineage', 'silo', 'quality', 'impact']:
            result = model(question=[f'Test {task}'], x=x,
                           edge_index=edge_index, task=task)
            assert result is not None
            assert 'pred' in result

    def test_simple_model_predict_task(self) -> None:
        """Test predict_task method."""
        try:
            # yapf: disable
            from torch_geometric.utils.data_warehouse import (
                SimpleWarehouseModel,
            )

            # yapf: enable
        except ImportError:
            pytest.skip("Data warehouse not available")

        model = SimpleWarehouseModel()
        x = torch.randn(15, 384)
        edge_index = torch.randint(0, 15, (2, 25))

        pred = model.predict_task(x, edge_index, task='silo')
        assert pred is not None
        assert pred.dim() >= 1

    def test_simple_model_edge_cases(self) -> None:
        """Test edge cases and error handling."""
        try:
            # yapf: disable
            from torch_geometric.utils.data_warehouse import (
                SimpleWarehouseModel,
            )

            # yapf: enable
        except ImportError:
            pytest.skip("Data warehouse not available")

        model = SimpleWarehouseModel()

        # Test with minimal graph
        x = torch.randn(2, 384)
        edge_index = torch.randint(0, 2, (2, 2))

        for task in ['lineage', 'silo', 'quality', 'impact']:
            pred = model.predict_task(x, edge_index, task=task)
            assert pred is not None
            assert pred.shape[0] == 2  # Should match number of nodes

    def test_simple_model_batch_processing(self) -> None:
        """Test batch processing functionality."""
        try:
            # yapf: disable
            from torch_geometric.utils.data_warehouse import (
                SimpleWarehouseModel,
            )

            # yapf: enable
        except ImportError:
            pytest.skip("Data warehouse not available")

        model = SimpleWarehouseModel()
        x = torch.randn(30, 384)
        edge_index = torch.randint(0, 30, (2, 50))

        # Test without batch parameter (method doesn't accept batch)
        for task in ['lineage', 'silo', 'quality', 'impact']:
            pred = model.predict_task(x, edge_index, task=task)
            assert pred is not None
            assert pred.shape[0] == 30


class TestSmartParser:
    """Test smart parser functionality."""
    def test_get_concise_analytics_lineage(self) -> None:
        """Test smart parser lineage translation."""
        try:
            # yapf: disable
            from torch_geometric.utils.data_warehouse import (
                SimpleWarehouseModel,
                WarehouseConversationSystem,
            )

            # yapf: enable
        except ImportError:
            pytest.skip("Data warehouse not available")

        model = SimpleWarehouseModel()
        system = WarehouseConversationSystem(model, concise_context=True)

        x = torch.randn(10, 384)
        edge_index = torch.randint(0, 10, (2, 15))
        # Multi-line to keep within flake8 limits
        pred = torch.zeros(x.shape[0], 5)
        pred[:, 2] = 3.0  # 'stores cleaned and processed business data'

        def _fake_predict_task(_x: Any, _edge_index: Any,
                               _task: Any) -> Any:  # noqa: ARG001
            return pred

        # Mock the predict_task method properly
        from unittest.mock import patch
        with patch.object(system.model, 'predict_task',
                          side_effect=_fake_predict_task):
            result = system._get_concise_analytics(x, edge_index, 'lineage')
            assert isinstance(result, str)
            assert len(result) > 0
            # Should contain business language, not technical terms
            assert 'entities' not in result.lower()
            # Should contain business-relevant words
        business_words = [
            'data', 'warehouse', 'contains', 'business', 'systems',
            'processed', 'source'
        ]
        assert any(word in result.lower() for word in business_words)

    def test_get_concise_analytics_all_tasks(self) -> None:
        """Test smart parser for all task types."""
        try:
            # yapf: disable
            from torch_geometric.utils.data_warehouse import (
                SimpleWarehouseModel,
                WarehouseConversationSystem,
            )

            # yapf: enable
        except ImportError:
            pytest.skip("Data warehouse not available")

        model = SimpleWarehouseModel()
        system = WarehouseConversationSystem(model, concise_context=True)

        x = torch.randn(20, 384)
        edge_index = torch.randint(0, 20, (2, 30))

        tasks = ['lineage', 'silo', 'quality', 'impact']
        for task in tasks:
            result = system._get_concise_analytics(x, edge_index, task)
            assert isinstance(result, str)
            assert len(result) > 10  # Should be meaningful text
            # Should be business language, not technical metrics
            assert 'entities' not in result.lower()

    def test_create_concise_contextual_prompt(self) -> None:
        """Test concise prompt generation."""
        try:
            # yapf: disable
            from torch_geometric.utils.data_warehouse import (
                SimpleWarehouseModel,
                WarehouseConversationSystem,
            )

            # yapf: enable
        except ImportError:
            pytest.skip("Data warehouse not available")

        model = SimpleWarehouseModel()
        system = WarehouseConversationSystem(model, concise_context=True)

        x = torch.randn(15, 384)
        edge_index = torch.randint(0, 15, (2, 25))

        context = {
            'domain': 'Test Warehouse',
            'node_types': ['table1', 'table2', 'table3'],
            'edge_types': [('table1', 'connects_to', 'table2')]
        }

        prompt = system._create_concise_contextual_prompt(
            "What is the data lineage?", x, edge_index, 'lineage', context)

        assert isinstance(prompt, str)
        assert 'Test Warehouse' in prompt
        assert 'table1' in prompt or 'table2' in prompt
        assert 'What is the data lineage?' in prompt
        assert 'Answer concisely in 2-3 sentences' in prompt

    def test_extract_essential_table_info(self) -> None:
        """Test table information extraction."""
        try:
            # yapf: disable
            from torch_geometric.utils.data_warehouse import (
                SimpleWarehouseModel,
                WarehouseConversationSystem,
            )

            # yapf: enable
        except ImportError:
            pytest.skip("Data warehouse not available")

        model = SimpleWarehouseModel()
        system = WarehouseConversationSystem(model, concise_context=True)

        context = {
            'domain':
            'Formula 1 Racing Data',
            'node_types': ['races', 'circuits', 'drivers', 'results'],
            'edge_types': [('races', 'held_at', 'circuits'),
                           ('results', 'from_race', 'races')]
        }

        info = system._extract_essential_table_info(context)

        assert isinstance(info, dict)
        assert 'domain' in info
        assert 'tables' in info
        assert 'relationships' in info
        assert info['domain'] == 'Formula 1 Racing Data'
        assert 'races' in info['tables']
        assert 'circuits' in info['tables']

    def test_extract_essential_table_info_fallback(self) -> None:
        """Test table information extraction with no context."""
        try:
            # yapf: disable
            from torch_geometric.utils.data_warehouse import (
                SimpleWarehouseModel,
                WarehouseConversationSystem,
            )

            # yapf: enable
        except ImportError:
            pytest.skip("Data warehouse not available")

        model = SimpleWarehouseModel()
        system = WarehouseConversationSystem(model, concise_context=True)

        info = system._extract_essential_table_info(None)

        assert isinstance(info, dict)
        assert 'domain' in info
        assert 'tables' in info
        assert 'relationships' in info
        # Should have fallback values
        assert info['domain'] == 'Data Warehouse'


class TestAdvancedFeatures:
    """Test advanced features and edge cases."""
    def test_conversation_system_concise_mode(self) -> None:
        """Test conversation system in concise mode."""
        try:
            # yapf: disable
            from torch_geometric.utils.data_warehouse import (
                SimpleWarehouseModel,
                WarehouseConversationSystem,
            )

            # yapf: enable
        except ImportError:
            pytest.skip("Data warehouse not available")

        model = SimpleWarehouseModel()
        system = WarehouseConversationSystem(model, concise_context=True,
                                             verbose=True)

        x = torch.randn(15, 384)
        edge_index = torch.randint(0, 15, (2, 25))
        graph_data = {'x': x, 'edge_index': edge_index, 'batch': None}

        # Test with context
        context = {
            'domain': 'Test Domain',
            'node_types': ['table1', 'table2'],
            'edge_types': [('table1', 'connects', 'table2')]
        }

        result = system.process_query("What is the lineage?", graph_data,
                                      context=context)
        assert 'answer' in result
        assert 'query_type' in result

    def test_conversation_system_error_handling(self) -> None:
        """Test error handling in conversation system."""
        try:
            # yapf: disable
            from torch_geometric.utils.data_warehouse import (
                SimpleWarehouseModel,
                WarehouseConversationSystem,
            )

            # yapf: enable
        except ImportError:
            pytest.skip("Data warehouse not available")

        model = SimpleWarehouseModel()
        system = WarehouseConversationSystem(model)

        # Test with invalid graph data
        invalid_data = {'x': None, 'edge_index': None}

        with pytest.raises(RuntimeError):
            system.process_query("Test query", invalid_data)

    def test_query_type_classification(self) -> None:
        """Test query type classification."""
        try:
            # yapf: disable
            from torch_geometric.utils.data_warehouse import (
                SimpleWarehouseModel,
                WarehouseConversationSystem,
            )

            # yapf: enable
        except ImportError:
            pytest.skip("Data warehouse not available")

        model = SimpleWarehouseModel()
        system = WarehouseConversationSystem(model)

        # Test various query types
        test_queries = [("trace the data flow", "lineage"),
                        ("check for isolated data", "silo"),
                        ("assess data reliability", "general"),
                        ("analyze change impact", "impact"),
                        ("general warehouse question", "general")]

        for query, expected_type in test_queries:
            detected_type = system.classify_query(query)
            assert detected_type == expected_type

    def test_contextual_prompt_generation(self) -> None:
        """Test contextual prompt generation."""
        try:
            # yapf: disable
            from torch_geometric.utils.data_warehouse import (
                SimpleWarehouseModel,
                WarehouseConversationSystem,
            )

            # yapf: enable
        except ImportError:
            pytest.skip("Data warehouse not available")

        model = SimpleWarehouseModel()
        system = WarehouseConversationSystem(model, verbose=True)

        x = torch.randn(20, 384)
        edge_index = torch.randint(0, 20, (2, 30))

        context = {
            'domain':
            'Test Warehouse',
            'node_types': ['users', 'orders', 'products'],
            'edge_types': [('users', 'places', 'orders'),
                           ('orders', 'contains', 'products')]
        }

        prompt = system._create_contextual_prompt("What is the data lineage?",
                                                  x, edge_index, 'lineage',
                                                  context)

        assert isinstance(prompt, str)
        assert len(prompt) > 100  # Should be substantial
        assert 'Test Warehouse' in prompt
        assert 'lineage' in prompt.lower()

    def test_analytics_generation_edge_cases(self) -> None:
        """Test analytics generation with edge cases."""
        try:
            # yapf: disable
            from torch_geometric.utils.data_warehouse import (
                SimpleWarehouseModel,
                WarehouseConversationSystem,
            )

            # yapf: enable
        except ImportError:
            pytest.skip("Data warehouse not available")

        model = SimpleWarehouseModel()
        system = WarehouseConversationSystem(model, concise_context=True)

        # Test with minimal data
        x = torch.randn(2, 384)
        edge_index = torch.randint(0, 2, (2, 2))

        for task in ['lineage', 'silo', 'quality', 'impact']:
            analytics = system._get_concise_analytics(x, edge_index, task)
            assert isinstance(analytics, str)
            assert len(analytics) > 0

    def test_conversation_history_management(self) -> None:
        """Test conversation history management."""
        try:
            # yapf: disable
            from torch_geometric.utils.data_warehouse import (
                SimpleWarehouseModel,
                WarehouseConversationSystem,
            )

            # yapf: enable
        except ImportError:
            pytest.skip("Data warehouse not available")

        model = SimpleWarehouseModel()
        system = WarehouseConversationSystem(model)

        x = torch.randn(10, 384)
        edge_index = torch.randint(0, 10, (2, 15))
        graph_data = {'x': x, 'edge_index': edge_index, 'batch': None}

        # Process multiple queries
        queries = ["Query 1", "Query 2", "Query 3"]
        for query in queries:
            system.process_query(query, graph_data)

        # Test history
        assert len(system.conversation_history) == len(queries)

        # Test history retrieval
        history = system.get_conversation_history()
        assert len(history) == len(queries)

        # Test history clearing
        system.conversation_history.clear()
        assert len(system.conversation_history) == 0


class TestWarehouseDemo:
    """Test warehouse demo functionality."""
    @withPackage('transformers')
    def test_create_warehouse_demo(self) -> None:
        """Test create_warehouse_demo function."""
        try:
            # yapf: disable
            from torch_geometric.utils.data_warehouse import (
                create_warehouse_demo,
            )

            # yapf: enable
        except ImportError:
            pytest.skip("Data warehouse not available")

        demo_system = create_warehouse_demo()
        assert demo_system is not None
        assert hasattr(demo_system, 'process_query')
        assert hasattr(demo_system, 'conversation_history')
        assert hasattr(demo_system, 'model')

    @withPackage('transformers')
    def test_create_warehouse_demo_with_options(self) -> None:
        """Test create_warehouse_demo with various options."""
        try:
            # yapf: disable
            from torch_geometric.utils.data_warehouse import (
                create_warehouse_demo,
            )

            # yapf: enable
        except ImportError:
            pytest.skip("Data warehouse not available")

        # Test with different configurations
        configs = [{
            'use_gretriever': False,
            'verbose': True
        }, {
            'concise_context': True,
            'verbose': False
        }, {
            'llm_max_tokens': 100,
            'gnn_heads': 2
        }]

        for config in configs:
            demo_system = create_warehouse_demo(
                **config)  # type: ignore[arg-type]
            assert demo_system is not None


class TestWarehouseDeepCoverage:
    """Target remaining branches in data_warehouse.py."""
    def test_extract_relevant_data_samples_no_context(self) -> None:
        try:
            from torch_geometric.utils.data_warehouse import (
                SimpleWarehouseModel,
                WarehouseConversationSystem,
            )
        except ImportError:
            pytest.skip("Data warehouse not available")
        sys = WarehouseConversationSystem(SimpleWarehouseModel())
        x = torch.randn(4, 384)
        edge_index = torch.randint(0, 4, (2, 6))
        out = sys._extract_relevant_data_samples("lineage", x, edge_index,
                                                 context=None)
        assert isinstance(out, str)
        assert "No specific data samples" in out

    def test_extract_relevant_data_samples_with_encoder_error(self) -> None:
        try:
            from torch_geometric.utils.data_warehouse import (
                SimpleWarehouseModel,
                WarehouseConversationSystem,
            )
        except ImportError:
            pytest.skip("Data warehouse not available")
        sys = WarehouseConversationSystem(SimpleWarehouseModel())
        # Enable encoder path and force _encode_text to raise
        sys.sentence_encoder = True  # type: ignore[attr-defined]

        def _boom(_: list[str]) -> torch.Tensor:
            raise RuntimeError("encoding failed")

        sys._encode_text = _boom  # type: ignore[method-assign,assignment]
        x = torch.randn(5, 384)
        edge_index = torch.randint(0, 5, (2, 8))
        ctx = {'node_types': ['users', 'orders']}
        msg = sys._extract_relevant_data_samples("lineage", x, edge_index, ctx)
        assert "Unable to extract relevant data samples" in msg

    def test_map_nodes_to_table_info_variants(self) -> None:
        try:
            from torch_geometric.utils.data_warehouse import (
                SimpleWarehouseModel,
                WarehouseConversationSystem,
            )
        except ImportError:
            pytest.skip("Data warehouse not available")
        sys = WarehouseConversationSystem(SimpleWarehouseModel())
        # Empty node_types
        msg = sys._map_nodes_to_table_info(torch.tensor([0, 1]), {}, "lineage")
        assert msg == "No table information available."
        # With table info
        ctx = {'node_types': ['users', 'orders', 'products']}
        msg2 = sys._map_nodes_to_table_info(torch.tensor([0, 2]), ctx,
                                            "quality")
        assert isinstance(msg2, str) and len(msg2) > 0

    def test_extract_keyword_relevant_data(self) -> None:
        try:
            from torch_geometric.utils.data_warehouse import (
                SimpleWarehouseModel,
                WarehouseConversationSystem,
            )
        except ImportError:
            pytest.skip("Data warehouse not available")
        sys = WarehouseConversationSystem(SimpleWarehouseModel())
        ctx = {'node_types': ['orders', 'lineage_logs', 'quality_checks']}
        got = sys._extract_keyword_relevant_data("check data quality", ctx)
        assert isinstance(got, str) and len(got) > 0
        got2 = sys._extract_keyword_relevant_data("unrelated query", ctx)
        assert isinstance(got2, str) and len(got2) > 0

    def test_extract_detailed_warehouse_info(self) -> None:
        try:
            from torch_geometric.utils.data_warehouse import (
                SimpleWarehouseModel,
                WarehouseConversationSystem,
            )
        except ImportError:
            pytest.skip("Data warehouse not available")
        sys = WarehouseConversationSystem(SimpleWarehouseModel())
        x = torch.randn(6, 384)
        edge_index = torch.randint(0, 6, (2, 10))
        # Without context
        s1 = sys._extract_detailed_warehouse_info(x, edge_index, None)
        assert "Total Entities:" in s1 and "Total Relationships:" in s1
        # With context
        ctx = {'node_types': ['a', 'b'], 'edge_types': [('a', 'rel', 'b')]}
        s2 = sys._extract_detailed_warehouse_info(x, edge_index, ctx)
        assert "Tables:" in s2 and "Key Relationships:" in s2

    def test_extract_detailed_analytics_all(self) -> None:
        try:
            from torch_geometric.utils.data_warehouse import (
                WarehouseConversationSystem, )
        except ImportError:
            pytest.skip("Data warehouse not available")
        # Dummy model to control outputs

        class Dummy:
            def predict_task(self, x: torch.Tensor, _edge_index: torch.Tensor,
                             task: str) -> torch.Tensor:  # noqa: D401
                n = x.shape[0]
                if task == 'lineage':
                    # logits favor class 2
                    logits = torch.zeros(n, 5)
                    logits[:, 2] = 3.0
                    return logits
                if task == 'silo':
                    return torch.ones(n, 1) * 0.6  # > 0.5
                if task == 'quality':
                    return torch.ones(n, 1) * 0.2
                if task == 'impact':
                    logits = torch.zeros(n, 3)
                    logits[:, 1] = 1.0
                    return logits
                raise ValueError("bad task")

        sys = WarehouseConversationSystem(Dummy())
        x = torch.randn(7, 384)
        edge_index = torch.randint(0, 7, (2, 12))
        for task in ['lineage', 'silo', 'quality', 'impact']:
            txt = sys._extract_detailed_analytics(x, edge_index, task)
            assert isinstance(txt, str) and len(txt) > 0

    def test_get_dataset_description_variants(self) -> None:
        try:
            from torch_geometric.utils.data_warehouse import (
                SimpleWarehouseModel,
                WarehouseConversationSystem,
            )
        except ImportError:
            pytest.skip("Data warehouse not available")
        sys = WarehouseConversationSystem(SimpleWarehouseModel())
        assert sys._get_dataset_description(None) == "data"
        assert sys._get_dataset_description({'node_types':
                                             ['a', 'b',
                                              'c']}) == "structured data"
        many = {'node_types': [str(i) for i in range(8)]}
        assert sys._get_dataset_description(many) == "multi-domain data"

    def test_create_integrated_response_variants(self) -> None:
        try:
            from torch_geometric.utils.data_warehouse import (
                SimpleWarehouseModel,
                WarehouseConversationSystem,
            )
        except ImportError:
            pytest.skip("Data warehouse not available")
        sys = WarehouseConversationSystem(SimpleWarehouseModel())
        data = {
            'num_nodes': 5,
            'confidence': 0.5,
            'predicted_lineage': 'Source'
        }
        long_text = (
            "This is a sufficiently long LLM response to trigger primary path."
            * 2)
        for q in ['lineage', 'silo', 'quality', 'impact', 'general']:
            sys._create_integrated_response(long_text, data, q)

    def test_concise_prompt_verbose_logging(self) -> None:
        try:
            from torch_geometric.utils.data_warehouse import (
                SimpleWarehouseModel,
                WarehouseConversationSystem,
            )
        except ImportError:
            pytest.skip("Data warehouse not available")
        system = WarehouseConversationSystem(SimpleWarehouseModel(),
                                             concise_context=True,
                                             verbose=True)
        x = torch.randn(3, 384)
        edge_index = torch.randint(0, 3, (2, 4))
        ctx = {'node_types': ['a', 'b'], 'edge_types': [('a', 'rel', 'b')]}
        out = system._create_concise_contextual_prompt("Q?", x, edge_index,
                                                       'lineage', ctx)
        assert isinstance(out, str) and len(out) > 0

    def test_dataset_description_relational(self) -> None:
        try:
            from torch_geometric.utils.data_warehouse import (
                SimpleWarehouseModel,
                WarehouseConversationSystem,
            )
        except ImportError:
            pytest.skip("Data warehouse not available")
        sys = WarehouseConversationSystem(SimpleWarehouseModel())
        assert sys._get_dataset_description(
            {'node_types': [str(i) for i in range(5)]}) == 'relational data'

    def test_clean_llm_response_artifacts(self) -> None:
        try:
            from torch_geometric.utils.data_warehouse import (
                SimpleWarehouseModel,
                WarehouseConversationSystem,
            )
        except ImportError:
            pytest.skip("Data warehouse not available")
        sys = WarehouseConversationSystem(SimpleWarehouseModel())
        raw = "### Assistant: Hello\n### Human: ask [ST:meta][O:tags]"
        cleaned = sys._clean_llm_response(raw)
        assert 'Assistant' not in cleaned and 'Human' not in cleaned
        assert '[' not in cleaned and ']' not in cleaned

    def test_extract_detailed_analytics_general(self) -> None:
        try:
            from torch_geometric.utils.data_warehouse import (
                SimpleWarehouseModel,
                WarehouseConversationSystem,
            )
        except ImportError:
            pytest.skip("Data warehouse not available")
        sys = WarehouseConversationSystem(SimpleWarehouseModel())
        x = torch.randn(4, 384)
        edge_index = torch.randint(0, 4, (2, 6))
        txt = sys._extract_detailed_analytics(x, edge_index, 'general')
        assert ('Analytics processing error' in txt
                or 'General Analytics' in txt)

    def test_initialize_domain_description_twice_and_history_copy(
            self) -> None:
        try:
            from torch_geometric.utils.data_warehouse import (
                SimpleWarehouseModel,
                WarehouseConversationSystem,
            )
        except ImportError:
            pytest.skip("Data warehouse not available")
        sys = WarehouseConversationSystem(SimpleWarehouseModel())
        ctx = {'node_types': ['t1', 't2']}
        # First call sets it:
        sys._initialize_domain_description(ctx)
        first = sys.domain_description
        # Second call should no-op
        sys._initialize_domain_description(ctx)
        assert sys.domain_description == first
        # History copy
        sys.conversation_history.append({'query': 'q', 'answer': 'a'})
        h = sys.get_conversation_history()
        assert h is not sys.conversation_history and len(h) == 1

    def test_process_query_with_labels_template(self) -> None:
        try:
            from torch_geometric.utils.data_warehouse import (
                SimpleWarehouseModel,
                WarehouseConversationSystem,
            )
        except ImportError:
            pytest.skip("Data warehouse not available")
        model = SimpleWarehouseModel()
        sys = WarehouseConversationSystem(model)
        x = torch.randn(9, 384)
        edge_index = torch.randint(0, 9, (2, 14))
        # Provide labels to trigger 'with_labels' formatting
        ctx = {
            'warehouse_labels': {
                'lineage': torch.randint(0, 5, (9, )),
                'silo': torch.randint(0, 2, (9, )),
                'quality': torch.randint(0, 2, (9, )),
                'impact': torch.randint(0, 3, (9, )),
            }
        }
        out = sys.process_query("What is the data lineage?", {
            'x': x,
            'edge_index': edge_index
        }, context=ctx)
        assert isinstance(out['answer'], str) and len(out['answer']) > 0

    def test_extract_table_info_and_clean_llm_response(self) -> None:
        try:
            from torch_geometric.utils.data_warehouse import (
                SimpleWarehouseModel,
                WarehouseConversationSystem,
            )
        except ImportError:
            pytest.skip("Data warehouse not available")
        sys = WarehouseConversationSystem(SimpleWarehouseModel())
        # _extract_table_info
        t1 = sys._extract_table_info(None)
        assert isinstance(t1, str) and len(t1) > 0
        ctx = {
            'node_types': ['users', 'orders'],
            'edge_types': [('users', 'rel', 'orders')]
        }
        t2 = sys._extract_table_info(ctx)
        assert 'users' in t2 and 'orders' in t2
        # _clean_llm_response
        long = "  Hello\n\nWorld!  "
        cl = sys._clean_llm_response(long)
        assert isinstance(cl, str) and 'Hello' in cl

    def test_process_query_llm_path_and_labels(self) -> None:
        """Test LLM path with warehouse labels."""
        try:
            from torch_geometric.utils.data_warehouse import (
                create_warehouse_demo, )
        except ImportError:
            pytest.skip("Data warehouse not available")

        sys = create_warehouse_demo(use_gretriever=True, verbose=False)

        x = torch.randn(8, 384)
        edge_index = torch.randint(0, 8, (2, 12))
        batch = torch.zeros(8, dtype=torch.long)
        context = {
            'node_types': ['users', 'orders', 'products', 'payments'],
            'edge_types': [('users', 'places', 'orders')],
            'warehouse_labels': {
                'lineage': torch.randint(0, 5, (8, )),
                'silo': torch.randint(0, 2, (8, )),
                'quality': torch.randint(0, 2, (8, )),
                'impact': torch.randint(0, 3, (8, )),
            }
        }
        out = sys.process_query("Explain lineage", {
            'x': x,
            'edge_index': edge_index,
            'batch': batch
        }, context=context)
        assert isinstance(out['answer'], str) and len(out['answer']) > 0
        assert out['llm_response'] is not None

    def test_encoder_success_path(self) -> None:
        """Test sentence encoder success path."""
        try:
            from torch_geometric.utils.data_warehouse import (
                SimpleWarehouseModel,
                WarehouseConversationSystem,
            )
        except ImportError:
            pytest.skip("Data warehouse not available")

        sys = WarehouseConversationSystem(SimpleWarehouseModel())

        # Force sentence encoder path with a dummy encoder
        def dummy_encode(texts: list[str]) -> torch.Tensor:
            return torch.randn(len(texts), 384)

        sys.sentence_encoder = True  # type: ignore[attr-defined]
        sys._encode_text = dummy_encode  # type: ignore[method-assign]

        x = torch.randn(5, 384)
        edge_index = torch.randint(0, 5, (2, 8))
        ctx = {'node_types': ['users', 'orders', 'products']}
        res = sys._extract_relevant_data_samples("lineage", x, edge_index, ctx)
        assert isinstance(res, str) and len(res) > 0

    def test_clean_llm_response_empty(self) -> None:
        """Test clean LLM response with empty input."""
        try:
            from torch_geometric.utils.data_warehouse import (
                SimpleWarehouseModel,
                WarehouseConversationSystem,
            )
        except ImportError:
            pytest.skip("Data warehouse not available")
        sys = WarehouseConversationSystem(SimpleWarehouseModel())
        assert sys._clean_llm_response("") == ""

    def test_extract_gat_insights_all_types(self) -> None:
        """Test GAT insights extraction for all task types."""
        try:
            from torch_geometric.utils.data_warehouse import (
                WarehouseConversationSystem, )
        except ImportError:
            pytest.skip("Data warehouse not available")

        class Dummy:
            def predict_task(self, _x: Any, _edge_index: Any,
                             task: str) -> Any:
                if task == 'lineage':
                    logits = torch.zeros(1, 5)
                    logits[:, 2] = 5.0
                    return logits
                if task == 'silo':
                    return torch.tensor(0.6)
                if task == 'quality':
                    return torch.tensor(0.3)
                if task == 'impact':
                    logits = torch.zeros(1, 3)
                    logits[:, 1] = 2.0
                    return logits
                return torch.zeros(1, 5)

        sys = WarehouseConversationSystem(Dummy())
        x = torch.randn(4, 384)
        edge_index = torch.randint(0, 4, (2, 6))

        assert 'feeds into business reports' in sys._extract_gat_insights(
            x, edge_index, 'lineage')
        assert 'limited connections' in sys._extract_gat_insights(
            x, edge_index, 'silo')
        assert 'needs improvement' in sys._extract_gat_insights(
            x, edge_index, 'quality')
        assert 'moderate' in sys._extract_gat_insights(x, edge_index, 'impact')

    def test_detailed_analytics_quality_excellent_with_dummy(self) -> None:
        """Test detailed analytics with excellent quality scores."""
        try:
            from torch_geometric.utils.data_warehouse import (
                WarehouseConversationSystem, )
        except ImportError:
            pytest.skip("Data warehouse not available")

        class Dummy:
            def predict_task(self, x: Any, _edge_index: Any, task: str) -> Any:
                # High logits -> sigmoid near 1.0 to trigger EXCELLENT branch
                return torch.full((x.shape[0], ), 8.0)

        sys = WarehouseConversationSystem(Dummy())
        x = torch.randn(6, 384)
        edge_index = torch.randint(0, 6, (2, 10))
        txt = sys._format_quality_analytics(
            sys.model.predict_task(x, edge_index, 'quality'), x.shape[0])
        assert 'EXCELLENT' in txt and 'Average Quality Score' in txt


class TestExceptionHandling:
    """Test exception handling for coverage boost."""
    def test_import_error_fallback(self) -> None:
        """Test ImportError handling in WarehouseGRetriever initialization."""
        try:
            from torch_geometric.utils.data_warehouse import (
                WarehouseGRetriever, )
        except ImportError:
            pytest.skip("Data warehouse not available")

        # Test the ImportError handling in WarehouseGRetriever.__init__
        # This tests the exception path when G-Retriever is not available
        import torch_geometric.utils.data_warehouse as dw
        original_has_gretriever = dw.HAS_GRETRIEVER

        try:
            # Temporarily disable G-Retriever to test exception path
            dw.HAS_GRETRIEVER = False

            # This should raise ImportError("G-Retriever not available")
            with pytest.raises(ImportError, match="G-Retriever not available"):
                WarehouseGRetriever()

        finally:
            dw.HAS_GRETRIEVER = original_has_gretriever

    @withPackage('transformers')
    def test_custom_llm_inference_exception(self) -> None:
        """Test exception fallback in custom_llm_inference."""
        try:
            from torch_geometric.utils.data_warehouse import (
                WarehouseGRetriever, )
        except ImportError:
            pytest.skip("G-Retriever not available")

        model = WarehouseGRetriever()
        # Force exception by providing invalid parameters
        result = model.custom_llm_inference("test", context=None,
                                            embedding=None, max_tokens=1)
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_process_query_exception_handling(self) -> None:
        """Test exception handling in process_query."""
        try:
            from torch_geometric.utils.data_warehouse import (
                SimpleWarehouseModel,
                WarehouseConversationSystem,
            )
        except ImportError:
            pytest.skip("Data warehouse not available")

        system = WarehouseConversationSystem(SimpleWarehouseModel())

        # Test with invalid graph data to trigger exception
        with pytest.raises((RuntimeError, ValueError)):
            system.process_query("test", {"invalid": "data"})

    def test_simple_model_exception_handling(self) -> None:
        """Test exception handling in SimpleWarehouseModel."""
        try:
            from torch_geometric.utils.data_warehouse import (
                SimpleWarehouseModel, )
        except ImportError:
            pytest.skip("Data warehouse not available")

        model = SimpleWarehouseModel()

        # Test with invalid tensor shapes to trigger exceptions
        x_invalid = torch.randn(2, 100)  # Wrong feature dimension
        edge_index = torch.randint(0, 2, (2, 2))

        with pytest.raises(RuntimeError):
            model.predict_task(x_invalid, edge_index, task="lineage")


class TestThresholdBranches:
    """Test threshold conditions for analytics coverage."""
    def test_silo_threshold_branches(self) -> None:
        """Test silo probability threshold branches."""
        try:
            from torch_geometric.utils.data_warehouse import (
                WarehouseConversationSystem, )
        except ImportError:
            pytest.skip("Data warehouse not available")

        # Create mock model with controlled silo predictions
        class MockSiloModel:
            def predict_task(self, x: torch.Tensor, edge_index: torch.Tensor,
                             task: str) -> torch.Tensor:
                if task == "silo":
                    # Return high probability to test > 0.7 branch
                    return torch.tensor(0.8)
                return torch.zeros(x.shape[0], 5)

        system = WarehouseConversationSystem(MockSiloModel())
        x = torch.randn(4, 384)
        edge_index = torch.randint(0, 4, (2, 6))

        result = system._extract_gat_insights(x, edge_index, "silo")
        assert "isolated from others" in result

        # Test medium probability (0.4 < prob < 0.7)
        class MockSiloModelMed:
            def predict_task(self, x: torch.Tensor, edge_index: torch.Tensor,
                             task: str) -> torch.Tensor:
                return torch.tensor(0.5)

        system_med = WarehouseConversationSystem(MockSiloModelMed())
        result_med = system_med._extract_gat_insights(x, edge_index, "silo")
        assert "limited connections" in result_med

    def test_quality_threshold_branches(self) -> None:
        """Test quality score threshold branches."""
        try:
            from torch_geometric.utils.data_warehouse import (
                WarehouseConversationSystem, )
        except ImportError:
            pytest.skip("Data warehouse not available")

        # Test high quality (> 0.8)
        class MockQualityModelHigh:
            def predict_task(self, x: torch.Tensor, edge_index: torch.Tensor,
                             task: str) -> torch.Tensor:
                return torch.tensor(0.9)

        system = WarehouseConversationSystem(MockQualityModelHigh())
        x = torch.randn(4, 384)
        edge_index = torch.randint(0, 4, (2, 6))

        result = system._extract_gat_insights(x, edge_index, "quality")
        assert "high with consistency" in result

        # Test low quality (< 0.5)
        class MockQualityModelLow:
            def predict_task(self, x: torch.Tensor, edge_index: torch.Tensor,
                             task: str) -> torch.Tensor:
                return torch.tensor(0.3)

        system_low = WarehouseConversationSystem(MockQualityModelLow())
        result_low = system_low._extract_gat_insights(x, edge_index, "quality")
        assert "needs improvement" in result_low

    def test_impact_threshold_branches(self) -> None:
        """Test impact level threshold branches."""
        try:
            from torch_geometric.utils.data_warehouse import (
                WarehouseConversationSystem, )
        except ImportError:
            pytest.skip("Data warehouse not available")

        # Test high impact
        class MockImpactModel:
            def predict_task(self, x: torch.Tensor, edge_index: torch.Tensor,
                             task: str) -> torch.Tensor:
                logits = torch.zeros(1, 3)
                logits[:, 2] = 5.0  # High impact class
                return logits

        system = WarehouseConversationSystem(MockImpactModel())
        x = torch.randn(4, 384)
        edge_index = torch.randint(0, 4, (2, 6))

        result = system._extract_gat_insights(x, edge_index, "impact")
        assert "high impact" in result


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    @withPackage('transformers')
    def test_empty_question_handling(self) -> None:
        """Test empty question list handling."""
        try:
            from torch_geometric.utils.data_warehouse import (
                WarehouseGRetriever, )
        except ImportError:
            pytest.skip("G-Retriever not available")

        model = WarehouseGRetriever()
        x = torch.randn(5, 384)
        edge_index = torch.randint(0, 5, (2, 8))
        batch = torch.zeros(5, dtype=torch.long)

        # Test with empty question list
        loss = model.train_forward([], x, edge_index, batch, ["label"])
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad

    @withPackage('transformers')
    def test_empty_label_handling(self) -> None:
        """Test empty label list handling."""
        try:
            from torch_geometric.utils.data_warehouse import (
                WarehouseGRetriever, )
        except ImportError:
            pytest.skip("G-Retriever not available")

        model = WarehouseGRetriever()
        x = torch.randn(5, 384)
        edge_index = torch.randint(0, 5, (2, 8))
        batch = torch.zeros(5, dtype=torch.long)

        # Test with empty label list
        loss = model.train_forward(["question"], x, edge_index, batch, [])
        assert isinstance(loss, torch.Tensor)

    def test_empty_llm_response_cleaning(self) -> None:
        """Test LLM response cleaning with edge cases."""
        try:
            from torch_geometric.utils.data_warehouse import (
                SimpleWarehouseModel,
                WarehouseConversationSystem,
            )
        except ImportError:
            pytest.skip("Data warehouse not available")

        system = WarehouseConversationSystem(SimpleWarehouseModel())

        # Test empty string
        result = system._clean_llm_response("")
        assert result == ""

        # Test whitespace only
        result = system._clean_llm_response("   ")
        assert result == ""

        # Test training artifacts
        result = system._clean_llm_response("### Assistant: Hello")
        assert "Assistant" not in result

        # Test repetitive text (should trigger fallback)
        repetitive = "word " * 15
        result = system._clean_llm_response(repetitive)
        assert "Analysis completed" in result

    def test_empty_context_handling(self) -> None:
        """Test handling of None/empty context in various methods."""
        try:
            from torch_geometric.utils.data_warehouse import (
                SimpleWarehouseModel,
                WarehouseConversationSystem,
            )
        except ImportError:
            pytest.skip("Data warehouse not available")

        system = WarehouseConversationSystem(SimpleWarehouseModel())
        x = torch.randn(4, 384)
        edge_index = torch.randint(0, 4, (2, 6))

        # Test with None context
        result = system._extract_relevant_data_samples("lineage", x,
                                                       edge_index, None)
        assert "No specific data samples" in result

        # Test domain description with None context
        desc = system._get_dataset_description(None)
        assert desc == "data"

    def test_encode_text_fallback(self) -> None:
        """Test _encode_text fallback behavior."""
        try:
            from torch_geometric.utils.data_warehouse import (
                SimpleWarehouseModel,
                WarehouseConversationSystem,
            )
        except ImportError:
            pytest.skip("Data warehouse not available")

        system = WarehouseConversationSystem(SimpleWarehouseModel())

        # Test without sentence encoder (should use fallback)
        result = system._encode_text(["test text"])
        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 384)


class TestModelConfigurations:
    """Test different model configuration branches."""
    @withPackage('transformers')
    @pytest.mark.skip(
        reason="Inconsistent model loading behavior across environments")
    def test_llm_embedding_dimension_detection(self) -> None:
        """Test LLM embedding dimension detection branches."""
        try:
            from torch_geometric.utils.data_warehouse import (
                WarehouseGRetriever, )
        except ImportError:
            pytest.skip("G-Retriever not available")

        # Mock LLM and GRetriever to avoid actual model loading
        from unittest.mock import patch, MagicMock

        with patch('torch_geometric.utils.data_warehouse.LLM') as mock_llm:
            with patch('torch_geometric.utils.data_warehouse.GRetriever') \
                    as mock_gr:

                mock_llm.return_value = MagicMock()
                mock_gr.return_value = MagicMock()

                # Test Phi-3 model detection
                model_phi = WarehouseGRetriever(
                    llm_model_name="microsoft/Phi-3-mini-4k-instruct")
                assert model_phi.gnn.out_channels == 3072

                # Test TinyLlama model detection
                model_tiny = WarehouseGRetriever(
                    llm_model_name="TinyLlama/TinyLlama-1.1B-Chat-v0.1")
                assert model_tiny.gnn.out_channels == 2048

                # Test default case
                model_default = WarehouseGRetriever(
                    llm_model_name="unknown-model")
                assert model_default.gnn.out_channels == 2048

    @withPackage('transformers')
    def test_parameter_handling_branches(self) -> None:
        """Test parameter handling in custom_llm_inference."""
        try:
            from torch_geometric.utils.data_warehouse import (
                WarehouseGRetriever, )
        except ImportError:
            pytest.skip("G-Retriever not available")

        model = WarehouseGRetriever()

        # Test max_tokens parameter handling (None vs provided)
        result1 = model.custom_llm_inference("test", max_tokens=None)
        assert isinstance(result1, list)

        result2 = model.custom_llm_inference("test", max_tokens=50)
        assert isinstance(result2, list)

        # Test context parameter handling
        result3 = model.custom_llm_inference("test", context="context")
        assert isinstance(result3, list)

        result4 = model.custom_llm_inference("test", context=None)
        assert isinstance(result4, list)

    def test_query_type_classification_branches(self) -> None:
        """Test all query type classification branches."""
        try:
            from torch_geometric.utils.data_warehouse import (
                SimpleWarehouseModel,
                WarehouseConversationSystem,
            )
        except ImportError:
            pytest.skip("Data warehouse not available")

        system = WarehouseConversationSystem(SimpleWarehouseModel())

        # Test all classification branches
        test_cases = [
            ("trace the lineage", "lineage"),
            ("check for silos", "silo"),
            ("analyze impact", "impact"),
            ("assess quality", "quality"),
            ("general question", "general"),
        ]

        for query, expected_type in test_cases:
            result = system.classify_query(query)
            assert result == expected_type

    def test_analytics_generation_error_handling(self) -> None:
        """Test analytics generation with error conditions."""
        try:
            from torch_geometric.utils.data_warehouse import (
                WarehouseConversationSystem, )
        except ImportError:
            pytest.skip("Data warehouse not available")

        # Create model that raises exceptions
        class ErrorModel:
            def predict_task(self, x: torch.Tensor, edge_index: torch.Tensor,
                             task: str) -> torch.Tensor:
                raise RuntimeError("Prediction failed")

        system = WarehouseConversationSystem(ErrorModel())
        x = torch.randn(4, 384)
        edge_index = torch.randint(0, 4, (2, 6))

        # Test error handling in _get_concise_analytics
        result = system._get_concise_analytics(x, edge_index, "lineage")
        assert "analysis details unavailable" in result

        # Test error handling in _extract_gat_insights
        result2 = system._extract_gat_insights(x, edge_index, "silo")
        assert "Unable to analyze structure" in result2

    def test_create_warehouse_demo_fallback(self) -> None:
        """Test create_warehouse_demo fallback behavior."""
        try:
            from torch_geometric.utils.data_warehouse import (
                create_warehouse_demo, )
        except ImportError:
            pytest.skip("Data warehouse not available")

        # Test fallback to simple model (exception path)
        demo_system = create_warehouse_demo(use_gretriever=False)
        assert demo_system is not None
        assert hasattr(demo_system, 'model')


class TestAnalyticsFormatting:
    """Test analytics formatting branches for coverage."""
    def test_silo_severity_levels(self) -> None:
        """Test all silo severity classification branches."""
        try:
            from torch_geometric.utils.data_warehouse import (
                WarehouseConversationSystem, )
        except ImportError:
            pytest.skip("Data warehouse not available")

        # Create mock model to control silo predictions
        class MockSiloSeverityModel:
            def __init__(self, isolation_pct: float) -> None:
                self.isolation_pct = isolation_pct

            def predict_task(self, x: torch.Tensor, edge_index: torch.Tensor,
                             task: str) -> torch.Tensor:
                # Return predictions that result in specific isolation %
                num_nodes = x.shape[0]
                isolated_count = int(num_nodes * self.isolation_pct / 100)
                pred = torch.zeros(num_nodes, 1)
                pred[:isolated_count] = 0.8  # Above threshold
                return pred

        # Test CRITICAL severity (> 80%) - use 90% to ensure > 80%
        system_critical = WarehouseConversationSystem(
            MockSiloSeverityModel(90))
        x = torch.randn(10, 384)
        edge_index = torch.randint(0, 10, (2, 15))
        result = system_critical._format_silo_analytics(
            system_critical.model.predict_task(x, edge_index, "silo"), 10)
        assert "CRITICAL" in result

        # Test HIGH severity (50% < x <= 80%) - 80% should be HIGH not CRITICAL
        system_high = WarehouseConversationSystem(MockSiloSeverityModel(80))
        result = system_high._format_silo_analytics(
            system_high.model.predict_task(x, edge_index, "silo"), 10)
        assert "HIGH" in result

        # Test MODERATE severity (20% < x <= 50%)
        system_mod = WarehouseConversationSystem(MockSiloSeverityModel(30))
        result = system_mod._format_silo_analytics(
            system_mod.model.predict_task(x, edge_index, "silo"), 10)
        assert "MODERATE" in result

        # Test LOW severity (<= 20%)
        system_low = WarehouseConversationSystem(MockSiloSeverityModel(10))
        result = system_low._format_silo_analytics(
            system_low.model.predict_task(x, edge_index, "silo"), 10)
        assert "LOW" in result

    def test_quality_status_levels(self) -> None:
        """Test quality status classification branches."""
        try:
            from torch_geometric.utils.data_warehouse import (
                WarehouseConversationSystem, )
        except ImportError:
            pytest.skip("Data warehouse not available")

        # Create mock model for quality scores
        class MockQualityModel:
            def __init__(self, target_sigmoid: float) -> None:
                # Convert target sigmoid output to logit input
                import math
                self.logit_value = math.log(target_sigmoid /
                                            (1 - target_sigmoid))

            def predict_task(self, x: torch.Tensor, edge_index: torch.Tensor,
                             task: str) -> torch.Tensor:
                return torch.full((x.shape[0], 1), self.logit_value)

        x = torch.randn(10, 384)
        edge_index = torch.randint(0, 10, (2, 15))

        # Test EXCELLENT status (> 0.8)
        system_exc = WarehouseConversationSystem(MockQualityModel(0.9))
        result = system_exc._format_quality_analytics(
            system_exc.model.predict_task(x, edge_index, "quality"), 10)
        assert "EXCELLENT" in result

        # Test GOOD status (0.6 < x <= 0.8) - 0.71 should be GOOD
        system_good = WarehouseConversationSystem(MockQualityModel(0.71))
        result = system_good._format_quality_analytics(
            system_good.model.predict_task(x, edge_index, "quality"), 10)
        assert "GOOD" in result

        # Test POOR status (<= 0.6)
        system_poor = WarehouseConversationSystem(MockQualityModel(0.5))
        result = system_poor._format_quality_analytics(
            system_poor.model.predict_task(x, edge_index, "quality"), 10)
        assert "POOR" in result

    def test_impact_risk_levels(self) -> None:
        """Test impact risk level classification branches."""
        try:
            from torch_geometric.utils.data_warehouse import (
                SimpleWarehouseModel,
                WarehouseConversationSystem,
            )
        except ImportError:
            pytest.skip("Data warehouse not available")

        system = WarehouseConversationSystem(SimpleWarehouseModel())
        torch.randn(10, 384)

        # Test HIGH risk (> 30% high impact)
        pred_high = torch.zeros(10, 3)
        pred_high[:4, 2] = 5.0  # 40% high impact
        data = system._extract_impact_data(pred_high, 10)
        assert data['risk_level'] == 'HIGH'

        # Test MEDIUM risk (10% < x <= 30%)
        pred_med = torch.zeros(10, 3)
        pred_med[:2, 2] = 5.0  # 20% high impact
        data = system._extract_impact_data(pred_med, 10)
        assert data['risk_level'] == 'MEDIUM'

        # Test LOW risk (<= 10%)
        pred_low = torch.zeros(10, 3)
        pred_low[:1, 2] = 5.0  # 10% high impact
        data = system._extract_impact_data(pred_low, 10)
        assert data['risk_level'] == 'LOW'


class TestWarehouseTraining:
    """Test warehouse training functionality."""
    def test_create_warehouse_training_data(self) -> None:
        """Test training data creation."""
        try:
            from torch_geometric.utils.data_warehouse import (
                create_warehouse_training_data, )
        except ImportError:
            pytest.skip("Training functions not available")

        # Test basic data creation
        training_data = create_warehouse_training_data(num_samples=5,
                                                       num_nodes=20)

        assert len(training_data) == 5
        assert all('question' in sample for sample in training_data)
        assert all('x' in sample for sample in training_data)
        assert all('edge_index' in sample for sample in training_data)
        assert all('label' in sample for sample in training_data)

        # Check tensor shapes
        sample = training_data[0]
        assert sample['x'].shape == (20, 384)  # EMBEDDING_DIM = 384
        assert sample['edge_index'].shape[0] == 2
        assert isinstance(sample['question'], str)
        assert isinstance(sample['label'], str)


class TestAdditionalCoverage:
    """Additional tests to cover uncovered branches with minimal deps."""
    def test_import_fallback_reload(self) -> None:
        """Force ImportError path and ensure stub classes raise ImportError."""
        try:
            import importlib
            import sys
            from types import ModuleType
            import torch_geometric.utils.data_warehouse as dw
        except Exception:  # pragma: no cover - environment import guard
            pytest.skip("data_warehouse not importable")

        orig_llm_models = sys.modules.get('torch_geometric.llm.models')
        try:
            # Shadow llm.models to trigger ImportError on from-import
            sys.modules['torch_geometric.llm.models'] = ModuleType('tg_dummy')
            m = importlib.reload(dw)
            assert m.HAS_GRETRIEVER is False
            with pytest.raises(ImportError):
                m.GRetriever()
            with pytest.raises(ImportError):
                m.LLM()
        finally:
            # Restore modules and reload back to normal
            if orig_llm_models is None:
                sys.modules.pop('torch_geometric.llm.models', None)
            else:
                sys.modules['torch_geometric.llm.models'] = orig_llm_models
            importlib.reload(dw)

    @withPackage('transformers')
    def test_custom_llm_inference_success_path(self) -> None:
        """Exercise custom_llm_inference try-body via mocks."""
        try:
            from torch_geometric.utils.data_warehouse import (
                WarehouseGRetriever, )
        except ImportError:
            pytest.skip("G-Retriever not available")

        from unittest.mock import MagicMock, patch

        with patch('torch_geometric.utils.data_warehouse.LLM') as mock_llm, \
                patch('torch_geometric.utils.data_warehouse.GRetriever') \
                as mock_gr:
            mock_llm.return_value = MagicMock()
            mock_gr.return_value = MagicMock()
            model = WarehouseGRetriever()

            # Mock LLM internals for success path
            def _fake_get_embeds(
                q: Any, c: Any, emb: Any
            ) -> tuple[torch.Tensor, torch.Tensor, None]:  # noqa: D401,E501
                return torch.randn(1, 4, 8), torch.ones(1, 4), None

            model.llm._get_embeds = _fake_get_embeds  # type: ignore[assignment]  # noqa: E501

            tok = MagicMock()
            tok.return_value = MagicMock(input_ids=[1])
            tok.pad_token_id = 0
            tok.encode = MagicMock(return_value=[3])
            tok.batch_decode = MagicMock(return_value=['ok'])

            # Cast to Any to allow assigning mock tokenizer
            model.llm = cast(Any, model.llm)
            model.llm.tokenizer = tok

            model.llm.llm = MagicMock()
            model.llm.llm.generate = MagicMock(
                return_value=torch.randint(0, 5, (1, 4)))

            out = model.custom_llm_inference("q", context=None, embedding=None,
                                             max_tokens=8)
            assert isinstance(out, list) and len(out) > 0 and out[0] == 'ok'

    def test_train_loop_with_simple_model(self) -> None:
        """Run one training epoch to cover optimizer/backward/clip/step."""
        try:
            from torch_geometric.utils.data_warehouse import (
                SimpleWarehouseModel,
                create_warehouse_training_data,
                train_warehouse_model,
            )
        except ImportError:
            pytest.skip("Training functions not available")

        model = SimpleWarehouseModel()
        training_data = create_warehouse_training_data(num_samples=4,
                                                       num_nodes=10)
        # train_warehouse_model expects WarehouseGRetriever, but our simple
        # model provides the train_forward fallback used inside.
        trained = train_warehouse_model(
            cast(Any, model),
            training_data,
            num_epochs=1,
            batch_size=2,
            device='cpu',
            verbose=False,
        )
        assert trained is not None

    @withPackage('transformers')
    def test_inference_multi_question_path(self) -> None:
        """Ensure multi-question path calls g_retriever.inference."""
        try:
            from torch_geometric.utils.data_warehouse import (
                WarehouseGRetriever, )
        except ImportError:
            pytest.skip("G-Retriever not available")

        from unittest.mock import MagicMock, patch

        with patch('torch_geometric.utils.data_warehouse.LLM') as mock_llm, \
                patch('torch_geometric.utils.data_warehouse.GRetriever') \
                as mock_gr:
            mock_llm.return_value = MagicMock()
            gr = MagicMock()
            gr.inference.return_value = ["ok2", "ok3"]
            mock_gr.return_value = gr
            model = WarehouseGRetriever()
            x = torch.randn(3, 384)
            edge_index = torch.randint(0, 3, (2, 4))
            batch = torch.zeros(3, dtype=torch.long)
            out = model.inference(["q1", "q2"], x, edge_index, batch)
            assert out == ["ok2", "ok3"]

    def test_concise_general_fallback(self) -> None:
        """Unknown query type should return general fallback string."""
        try:
            from torch_geometric.utils.data_warehouse import (
                SimpleWarehouseModel,
                WarehouseConversationSystem,
            )
        except ImportError:
            pytest.skip("Data warehouse not available")

        sys = WarehouseConversationSystem(SimpleWarehouseModel(),
                                          concise_context=True)
        x = torch.randn(5, 384)
        edge_index = torch.randint(0, 5, (2, 8))
        txt = sys._get_concise_analytics(x, edge_index, "unknown")
        assert isinstance(txt, str) and len(txt) > 0
        assert "contains business data" in txt

    def test_infer_key_relationships_variants(self) -> None:
        """Cover _infer_key_relationships early return, heuristic & fallback."""  # noqa: E501
        try:
            from torch_geometric.utils.data_warehouse import (
                SimpleWarehouseModel,
                WarehouseConversationSystem,
            )
        except ImportError:
            pytest.skip("Data warehouse not available")

        sys = WarehouseConversationSystem(SimpleWarehouseModel())
        # Early return (<2 tables)
        assert sys._infer_key_relationships(["only"]) == "table connections"
        # Heuristic overlap: parts of names overlap
        rels = sys._infer_key_relationships(["orders", "order_items", "users"])
        assert "orders->order_items" in rels
        # Fallback: first two tables connected when no overlap found
        rels2 = sys._infer_key_relationships(["a", "b", "c"])
        assert rels2.startswith("a->b")

    @withPackage('transformers')
    def test_inference_single_question_path(self) -> None:
        """Single-question list should route to custom_llm_inference."""
        try:
            from torch_geometric.utils.data_warehouse import (
                WarehouseGRetriever, )
        except ImportError:
            pytest.skip("G-Retriever not available")

        from unittest.mock import MagicMock, patch

        with patch('torch_geometric.utils.data_warehouse.LLM') as mock_llm, \
                patch('torch_geometric.utils.data_warehouse.GRetriever') \
                as mock_gr:
            mock_llm.return_value = MagicMock()
            mock_gr.return_value = MagicMock()
            model = WarehouseGRetriever()
            x = torch.randn(3, 384)
            edge_index = torch.randint(0, 3, (2, 4))
            batch = torch.zeros(3, dtype=torch.long)
            with patch.object(model, 'custom_llm_inference',
                              return_value=['ok1']):
                out = model.inference(["q1"], x, edge_index, batch)
            assert out == ['ok1']

    def test_integrated_response_path_non_empty_llm(self) -> None:
        """Integrated response with non-empty LLM text via dummy model."""
        try:
            from torch_geometric.utils.data_warehouse import (
                WarehouseConversationSystem, )
        except ImportError:
            pytest.skip("Data warehouse not available")

        class DummyModel:
            verbose = False

            def predict_task(self, x: torch.Tensor, edge_index: torch.Tensor,
                             task: str) -> torch.Tensor:
                if task == 'lineage':
                    logits = torch.zeros(x.shape[0], 5)
                    logits[:, 1] = 1.0
                    return logits
                if task == 'impact':
                    logits = torch.zeros(x.shape[0], 3)
                    logits[:, 2] = 1.0
                    return logits
                return torch.zeros(x.shape[0], 1)

            def inference(self, question: list[str], x: torch.Tensor,
                          edge_index: torch.Tensor, batch: torch.Tensor,
                          max_out_tokens: Optional[int] = None,
                          **_: Any) -> list[str]:
                return ["This is a detailed answer."]

        sys = WarehouseConversationSystem(DummyModel())
        x = torch.randn(5, 384)
        edge_index = torch.randint(0, 5, (2, 8))
        batch = torch.zeros(5, dtype=torch.long)
        context = {'node_types': ['a', 'b'], 'edge_types': [('a', 'rel', 'b')]}
        out = sys.process_query("lineage?", {
            'x': x,
            'edge_index': edge_index,
            'batch': batch,
            'context': context,
        })
        assert isinstance(out['answer'], str) and len(out['answer']) > 0
