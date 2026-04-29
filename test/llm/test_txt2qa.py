import asyncio
import os

import pytest

from torch_geometric.testing import onlyFullTest, withPackage


@withPackage('yaml', 'langgraph', 'jinja2', 'json_repair', 'openai', 'faiss')
class TestTxt2QAPureFunctions:
    """Tests for pure-logic / file-I/O functions that need no API key."""
    def test_load_input(self, tmp_path):
        from torch_geometric.llm.models.qa_gen import TaskStatus, load_input

        txt = tmp_path / "sample.txt"
        txt.write_text(
            "Graph neural networks are powerful tools for learning on graphs. "
            "They aggregate information from neighboring nodes. "
            "Message passing is the core mechanism. "
            "Each layer updates node representations. "
            "GNNs have been applied to social networks and molecules. "
            "PyTorch Geometric provides efficient GNN implementations. "
            "It supports various graph convolution operators. "
            "The library is widely used in research and industry. "
            "Scalability is achieved through mini-batch training. "
            "Recent advances include heterogeneous and temporal graphs.")

        state = {
            'task_id': 'test-load',
            'file_path': str(txt),
        }
        result = asyncio.get_event_loop().run_until_complete(load_input(state))

        assert result['status'] == TaskStatus.PROCESSING.value
        segments = result['segments']
        assert len(segments) > 0
        first = segments[0]
        assert 'text' in first
        assert 'start' in first
        assert 'end' in first
        assert 'chunk_id' in first
        assert first['chunk_id'] == 1

    def test_compute_metrics(self):
        from torch_geometric.llm.models.qa_gen import compute_metrics

        records = [
            {
                'question': 'What is a GNN?',
                'answer': 'A graph neural network.',
                'question_complexity': 'medium',
            },
            {
                'question': 'How does message passing work?',
                'answer': 'Nodes aggregate neighbor features.',
                'question_complexity': 'hard',
            },
        ]
        metrics = compute_metrics(records)

        assert metrics['total_pairs'] == 2
        assert 'average_question_length' in metrics
        assert 'average_answer_length' in metrics
        assert metrics['average_question_length'] > 0
        assert 'complexity_distribution' in metrics

    def test_route_after_validation_continue(self):
        from torch_geometric.llm.models.qa_gen import route_after_validation

        state = {'qa_pairs': [{'question': 'q', 'answer': 'a'}]}
        assert route_after_validation(state) == 'continue'

    def test_route_after_validation_feedback(self):
        from torch_geometric.llm.models.qa_gen import route_after_validation

        assert route_after_validation({'qa_pairs': []}) == 'feedback'
        assert route_after_validation({'qa_pairs': None}) == 'feedback'

    def test_should_retry(self):
        from torch_geometric.llm.models.qa_gen import TaskStatus, should_retry

        assert should_retry({'status': TaskStatus.RETRYING.value}) == 'retry'
        assert (should_retry({'status':
                              TaskStatus.FAILED.value}) == 'feedback_node')
        assert (should_retry({'status':
                              TaskStatus.COMPLETED.value}) == 'continue')


@onlyFullTest
@withPackage('yaml', 'langgraph', 'jinja2', 'json_repair', 'openai', 'faiss')
@pytest.mark.skipif(
    not os.getenv('NVIDIA_API_KEY'),
    reason='NVIDIA_API_KEY not set',
)
def test_generate_single_query_nim(tmp_path):
    """End-to-end smoke test: generate one QA pair via NIM."""
    from torch_geometric.llm.models.qa_gen import (
        LLMClient,
        TaskStatus,
        generate_qa_pairs,
        load_input,
    )

    txt = tmp_path / "sample.txt"
    txt.write_text(
        "Graph neural networks are powerful tools for learning on graphs. "
        "They aggregate information from neighboring nodes. "
        "Message passing is the core mechanism behind GNNs. "
        "Each layer updates node representations by combining features. "
        "GNNs have been successfully applied to social networks, "
        "molecular property prediction, and recommendation systems. "
        "PyTorch Geometric provides efficient GNN implementations "
        "with support for various graph convolution operators. "
        "The library is widely used in both research and industry. "
        "Scalability is achieved through mini-batch training on large graphs.")

    load_state = {
        'task_id': 'test-nim',
        'file_path': str(txt),
    }
    loaded = asyncio.get_event_loop().run_until_complete(
        load_input(load_state))
    segments = loaded['segments']
    assert len(segments) > 0

    client = LLMClient(
        generation_model='meta/llama-3.1-70b-instruct',
        backend='nim',
    )

    try:
        gen_state = {
            'task_id': 'test-nim',
            'file_path': str(txt),
            'segments': segments,
            'summary': [],
            'client': client,
            'num_pairs': 1,
            'parts': 1,
            'hard': False,
            'use_artifact': False,
            'question_only': False,
            'top_artifacts': None,
            'query_type_distribution':
            '{"multi_hop":0.4,"structural":0.3,"contextual":0.3}',
            'reasoning_type_distribution': None,
            'min_hops': 2,
            'max_hops': 4,
            'self_contained_question': False,
            'enable_hard_negatives': False,
            'status': TaskStatus.PENDING.value,
        }

        result = asyncio.get_event_loop().run_until_complete(
            generate_qa_pairs(gen_state))

        assert result['status'] != TaskStatus.FAILED.value, (
            f"generate_qa_pairs failed: {result.get('error_messages')}")
        qa_pairs = result.get('qa_pairs', [])
        assert len(qa_pairs) >= 1, "Expected at least 1 QA pair"

        pair = qa_pairs[0]
        assert 'question' in pair, "QA pair missing 'question'"
        assert 'answer' in pair, "QA pair missing 'answer'"
        assert len(pair['question']) > 0
        assert len(pair['answer']) > 0
    finally:
        client.cleanup()
