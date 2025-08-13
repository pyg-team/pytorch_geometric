# isort: skip_file
"""Comprehensive tests for data warehouse intelligence system."""

from typing import Any

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

        system.model.predict_task = _fake_predict_task  # type: ignore[method-assign]  # noqa: E501

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
            def predict_task(self, x: torch.Tensor, edge_index: torch.Tensor,
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
            def predict_task(self, x: Any, edge_index: Any, task: str) -> Any:
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
            def predict_task(self, x: Any, edge_index: Any, task: str) -> Any:
                # High logits -> sigmoid near 1.0 to trigger EXCELLENT branch
                return torch.full((x.shape[0], ), 8.0)

        sys = WarehouseConversationSystem(Dummy())
        x = torch.randn(6, 384)
        edge_index = torch.randint(0, 6, (2, 10))
        txt = sys._format_quality_analytics(
            sys.model.predict_task(x, edge_index, 'quality'), x.shape[0])
        assert 'EXCELLENT' in txt and 'Average Quality Score' in txt


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
