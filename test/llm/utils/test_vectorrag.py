import pytest
import torch

from torch_geometric.llm.utils.vectorrag import DocumentRetriever
from torch_geometric.testing import onlyRAG


@pytest.fixture
def sample_documents():
    """Fixture providing sample documents for testing."""
    return [
        "This is the first test document.",
        "This is the second test document.", "This is the third test document."
    ]


@pytest.fixture
def sample_model():
    """Fixture providing a mock model for testing."""
    from unittest.mock import Mock

    mock_model = Mock()
    # Mock the model to return a simple tensor when called
    mock_model.side_effect = [
        torch.zeros(1, 384),
        torch.ones(1, 384),
        torch.ones(1, 384) * 2,
        torch.ones(1, 384) * 1
    ]

    return mock_model


def test_save_load(sample_documents, sample_model, tmp_path):
    """Test whether saving/loading a DocumentRetriever maintains state."""
    retriever = DocumentRetriever(sample_documents, model=sample_model)
    retriever.save(tmp_path / "retriever.pth")
    loaded_retriever = DocumentRetriever.load(tmp_path / "retriever.pth",
                                              sample_model)
    assert retriever.raw_docs == loaded_retriever.raw_docs
    assert torch.allclose(retriever.embedded_docs,
                          loaded_retriever.embedded_docs)
    assert retriever.k_for_docs == loaded_retriever.k_for_docs
    assert retriever.model == loaded_retriever.model


@onlyRAG
def test_query(sample_documents, sample_model):
    """Test query functionality of DocumentRetriever."""
    retriever = DocumentRetriever(sample_documents, model=sample_model)
    query = "What is the first test document?"
    retrieved_docs = retriever.query(query)
    assert retrieved_docs == [sample_documents[0]]
