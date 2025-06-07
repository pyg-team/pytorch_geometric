import os
import random
import string

import pytest

from torch_geometric.datasets import WebQSPDataset
from torch_geometric.datasets.web_qsp_dataset import KGQABaseDataset
from torch_geometric.testing import (
    onlyFullTest,
    onlyOnline,
    onlyRAG,
    withPackage,
)


@pytest.mark.skip(reason="Times out")
@onlyOnline
@onlyFullTest
def test_web_qsp_dataset(tmp_path):
    dataset = WebQSPDataset(root=tmp_path)
    # Split for this dataset is 2826 train | 246 val | 1628 test
    # default split is train
    assert len(dataset) == 2826
    assert str(dataset) == "WebQSPDataset(2826)"

    dataset_train = WebQSPDataset(root=tmp_path, split="train")
    assert len(dataset_train) == 2826
    assert str(dataset_train) == "WebQSPDataset(2826)"

    dataset_val = WebQSPDataset(root=tmp_path, split="val")
    assert len(dataset_val) == 246
    assert str(dataset_val) == "WebQSPDataset(246)"

    dataset_test = WebQSPDataset(root=tmp_path, split="test")
    assert len(dataset_test) == 1628
    assert str(dataset_test) == "WebQSPDataset(1628)"


class MockSentenceTransformer:
    def __init__(self, *args, **kwargs):
        pass

    def to(self, device):
        return self

    def eval(self):
        return self

    def encode(self, sentences, batch_size=None, output_device=None):
        import torch

        def string_to_tensor(s: str) -> torch.Tensor:
            return torch.ones(1024).float()

        if isinstance(sentences, str):
            return string_to_tensor(sentences)
        return torch.stack([string_to_tensor(s) for s in sentences])


def create_mock_graphs(tmp_path: str, train_size: int, val_size: int,
                       test_size: int, num_nodes: int, num_edge_types: int,
                       num_trips: int, seed: int = 42):
    random.seed(seed)
    strkeys = string.ascii_letters + string.digits
    qa_strkeys = string.ascii_letters + string.digits + " "

    def create_mock_triplets(num_nodes: int, num_edges: int, num_trips: int):
        nodes = list(
            {"".join(random.sample(strkeys, 10))
             for i in range(num_nodes)})
        edges = list(
            {"".join(random.sample(strkeys, 10))
             for i in range(num_edges)})
        triplets = []

        for _ in range(num_trips):
            h = random.randint(0, num_nodes - 1)
            t = random.randint(0, num_nodes - 1)
            r = random.randint(0, num_edge_types - 1)
            triplets.append((nodes[h], edges[r], nodes[t]))
        return triplets

    train_triplets = [
        create_mock_triplets(num_nodes, num_edge_types, num_trips)
        for _ in range(train_size)
    ]
    val_triplets = [
        create_mock_triplets(num_nodes, num_edge_types, num_trips)
        for _ in range(val_size)
    ]
    test_triplets = [
        create_mock_triplets(num_nodes, num_edge_types, num_trips)
        for _ in range(test_size)
    ]

    train_questions = [
        "".join(random.sample(qa_strkeys, 10)) for _ in range(train_size)
    ]
    val_questions = [
        "".join(random.sample(qa_strkeys, 10)) for _ in range(val_size)
    ]
    test_questions = [
        "".join(random.sample(qa_strkeys, 10)) for _ in range(test_size)
    ]

    train_answers = [
        "".join(random.sample(qa_strkeys, 10)) for _ in range(train_size)
    ]
    val_answers = [
        "".join(random.sample(qa_strkeys, 10)) for _ in range(val_size)
    ]
    test_answers = [
        "".join(random.sample(qa_strkeys, 10)) for _ in range(test_size)
    ]

    train_graphs = {
        "graph": train_triplets,
        "question": train_questions,
        "answer": train_answers
    }
    val_graphs = {
        "graph": val_triplets,
        "question": val_questions,
        "answer": val_answers
    }
    test_graphs = {
        "graph": test_triplets,
        "question": test_questions,
        "answer": test_answers
    }

    from datasets import Dataset, DatasetDict, load_from_disk

    ds_train = Dataset.from_dict(train_graphs, split="train")
    ds_val = Dataset.from_dict(val_graphs, split="validation")
    ds_test = Dataset.from_dict(test_graphs, split="test")

    ds = DatasetDict({
        "train": ds_train,
        "validation": ds_val,
        "test": ds_test
    })

    def mock_load_dataset(name: str):
        # Save the dataset and then load it to emulate downloading from HF
        DATASET_CACHE_DIR = os.path.join(tmp_path,
                                         ".cache/huggingface/datasets", name)
        os.makedirs(DATASET_CACHE_DIR, exist_ok=True)

        ds.save_to_disk(DATASET_CACHE_DIR)
        dataset_remote = load_from_disk(DATASET_CACHE_DIR)
        return dataset_remote

    return mock_load_dataset, ds


@onlyRAG
@withPackage("datasets", "pandas")
def test_kgqa_base_dataset(tmp_path, monkeypatch):

    num_nodes = 500
    num_edge_types = 25
    num_trips = 5000

    # Mock the dataset graphs
    mock_load_dataset_func, expected_result = create_mock_graphs(
        tmp_path, train_size=10, val_size=5, test_size=5, num_nodes=num_nodes,
        num_edge_types=num_edge_types, num_trips=num_trips)

    import datasets

    monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset_func)

    # Mock the SentenceTransformer
    import torch_geometric.datasets.web_qsp_dataset
    monkeypatch.setattr(torch_geometric.datasets.web_qsp_dataset,
                        "SentenceTransformer", MockSentenceTransformer)

    dataset_train = KGQABaseDataset(root=tmp_path, dataset_name="TestDataset",
                                    split="train", use_pcst=False)
    assert len(dataset_train) == 10
    assert str(dataset_train) == "KGQABaseDataset(10)"
    for graph in dataset_train:
        assert graph.x.shape == (num_nodes, 1024)
        assert graph.edge_index.shape == (2, num_trips)
        assert graph.edge_attr.shape == (
            num_trips, 1024)  # Reminder: edge_attr encodes the entire triplet

    dataset_val = KGQABaseDataset(root=tmp_path, dataset_name="TestDataset",
                                  split="val", use_pcst=False)
    assert len(dataset_val) == 5
    assert str(dataset_val) == "KGQABaseDataset(5)"

    dataset_test = KGQABaseDataset(root=tmp_path, dataset_name="TestDataset",
                                   split="test", use_pcst=False)
    assert len(dataset_test) == 5
    assert str(dataset_test) == "KGQABaseDataset(5)"

    # TODO(zaristei): More rigorous tests to validate that values are correct
    # TODO(zaristei): Proper tests for PCST and CWQ
