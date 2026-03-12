import pytest
import torch

from torch_geometric.data import Data
from torch_geometric.llm.models import GLEM
from torch_geometric.llm.models.glem import deal_nan
from torch_geometric.loader import DataLoader, NeighborLoader
from torch_geometric.nn.models import GraphSAGE
from torch_geometric.testing import withPackage


def test_deal_nan_tensor_replaces_nans():
    x = torch.tensor([1.0, float('nan'), 3.0])
    result = deal_nan(x)

    expected = torch.tensor([1.0, 0.0, 3.0])
    assert torch.allclose(result, expected, equal_nan=True)
    assert isinstance(result, torch.Tensor)
    assert not torch.isnan(result).any()


def test_deal_nan_non_tensor_passthrough():
    assert deal_nan(42.0) == 42.0
    assert deal_nan("foo") == "foo"


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_deal_nan_tensor_dtypes(dtype):
    # Create a tensor with one NaN value
    x = torch.tensor([1.0, float('nan'), 3.0], dtype=dtype)
    result = deal_nan(x)

    expected = torch.tensor([1.0, 0.0, 3.0], dtype=dtype)

    # `bfloat16` doesn't support `allclose` directly on CPU,
    # so we cast to float32 for comparison
    if dtype == torch.bfloat16:
        assert torch.allclose(result.to(torch.float32),
                              expected.to(torch.float32), atol=1e-2)
    else:
        assert torch.allclose(result, expected, equal_nan=True)

    assert isinstance(result, torch.Tensor)
    assert not torch.isnan(result).any()
    assert result.dtype == dtype


def test_deal_nan_is_non_mutating():
    x = torch.tensor([1.0, float('nan'), 3.0])
    x_copy = x.clone()
    _ = deal_nan(x)
    assert torch.isnan(x).any()  # Original still contains NaN
    assert torch.allclose(x, x_copy, equal_nan=True)


@pytest.fixture
def tiny_graph_data():
    x = torch.randn(10, 16)  # 10 nodes, 16-dim features
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]],
                              dtype=torch.long)
    y = torch.randint(0, 3, (10, ))  # 3 classes
    is_gold = torch.tensor([True] * 5 + [False] * 5)  # 5 gold + 5 non-gold
    n_id = torch.arange(10)
    return Data(x=x, edge_index=edge_index, y=y, is_gold=is_gold, n_id=n_id)


@pytest.fixture
def dummy_text_data():
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.indices = torch.arange(10)
            self._data = type('obj', (), {'num_nodes': 10})()

        def __len__(self):
            return 10

        def __getitem__(self, idx):
            return {
                'input': {
                    'input_ids': torch.randint(100, 1000, (16, )),
                    'attention_mask': torch.ones(16, dtype=torch.long)
                },
                'labels': torch.tensor(idx % 3),
                'is_gold': torch.tensor(idx < 5),
                'n_id': torch.tensor(idx),
            }

    dataset = DummyDataset()
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    return loader


@pytest.fixture
def glem_model():
    gnn = GraphSAGE(
        in_channels=16,
        hidden_channels=32,
        num_layers=2,
        out_channels=3,
    )

    model = GLEM(lm_to_use='prajjwal1/bert-tiny', gnn_to_use=gnn,
                 out_channels=3, lm_use_lora=True, device='cpu')
    return model


@withPackage('transformers', 'sentencepiece', 'accelerate')
def test_glem_initialization(glem_model):
    assert glem_model.lm is not None
    assert glem_model.gnn is not None
    assert glem_model.lm.num_labels == 3


@withPackage('transformers', 'sentencepiece', 'accelerate', 'pyg_lib',
             'torch_sparse')
@pytest.mark.parametrize('is_augmented', [True, False])
def test_glem_pretrain(glem_model, tiny_graph_data, dummy_text_data,
                       is_augmented):
    # Test LM pretraining
    optimizer = torch.optim.Adam(glem_model.lm.parameters(), lr=1e-3)
    pseudo_labels = torch.randint(0, 3, (10, ))

    glem_model.pre_train_lm(
        train_loader=dummy_text_data,
        optimizer=optimizer,
        num_epochs=5,
        patience=1,
        ext_pseudo_labels=pseudo_labels,
        is_augmented=is_augmented,
        verbose=True,
    )

    # Test GNN pretraining
    loader = NeighborLoader(tiny_graph_data, num_neighbors=[2, 2],
                            batch_size=4, input_nodes=torch.arange(10))
    optimizer = torch.optim.Adam(glem_model.gnn.parameters(), lr=1e-3)
    pseudo_labels = torch.randint(0, 3, (10, ))

    glem_model.pre_train_gnn(
        train_loader=loader,
        optimizer=optimizer,
        num_epochs=5,
        patience=1,
        ext_pseudo_labels=pseudo_labels,
        is_augmented=is_augmented,
        verbose=True,
    )


@withPackage('transformers', 'sentencepiece', 'accelerate', 'pyg_lib',
             'torch_sparse')
@pytest.mark.parametrize('is_augmented', [True, False])
def test_glem_train(glem_model, tiny_graph_data, dummy_text_data,
                    is_augmented):
    # Test LM training
    optimizer = torch.optim.Adam(glem_model.lm.parameters(), lr=1e-3)
    pseudo_labels = torch.randint(0, 3, (10, ))

    acc, loss = glem_model.train(
        em_phase='lm',
        train_loader=dummy_text_data,
        optimizer=optimizer,
        epoch=1,
        pseudo_labels=pseudo_labels,
        is_augmented=is_augmented,
        verbose=True,
    )
    assert isinstance(acc, float) and isinstance(loss, float)
    assert 0 <= acc <= 1
    assert loss >= 0

    # Test GNN training
    loader = NeighborLoader(tiny_graph_data, num_neighbors=[2, 2],
                            batch_size=4, input_nodes=torch.arange(10))
    optimizer = torch.optim.Adam(glem_model.gnn.parameters(), lr=1e-3)
    pseudo_labels = torch.randint(0, 3, (10, ))

    acc, loss = glem_model.train(
        em_phase='gnn',
        train_loader=loader,
        optimizer=optimizer,
        epoch=1,
        pseudo_labels=pseudo_labels,
        is_augmented=is_augmented,
        verbose=True,
    )
    assert isinstance(acc, float) and isinstance(loss, float)
    assert 0 <= acc <= 1
    assert loss >= 0


@withPackage('transformers', 'sentencepiece', 'accelerate', 'pyg_lib',
             'torch_sparse')
def test_glem_inference(glem_model, tiny_graph_data, dummy_text_data):
    # Test LM inference
    preds = glem_model.inference('lm', dummy_text_data, verbose=True)
    assert preds.shape == (10, 3)  # 10 nodes, 3 classes
    assert not torch.isnan(preds).any()

    # Test GNN inference
    loader = NeighborLoader(tiny_graph_data, num_neighbors=[-1], batch_size=10,
                            input_nodes=torch.arange(10))

    preds = glem_model.inference('gnn', loader, verbose=True)
    assert preds.shape == (10, 3)
    assert not torch.isnan(preds).any()


@withPackage('transformers', 'sentencepiece', 'accelerate')
def test_glem_loss_function(glem_model):
    logits = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                          requires_grad=True)
    labels = torch.tensor([0, 1])
    is_gold = torch.tensor([True, False])
    pseudo_labels = torch.tensor([0, 2])

    loss_func = torch.nn.CrossEntropyLoss()

    # only gold
    loss1 = glem_model.loss(logits, labels, loss_func, is_gold, None, 0.5,
                            is_augmented=False)
    expected1 = loss_func(logits, labels)
    assert torch.allclose(loss1, expected1)

    # mix gold + pseudo
    loss2 = glem_model.loss(logits, labels, loss_func, is_gold, pseudo_labels,
                            0.3, is_augmented=True)
    mle = loss_func(logits[0:1], labels[0:1])  # gold part
    pseudo = loss_func(logits[1:2], pseudo_labels[1:2])  # pseudo part
    expected2 = 0.3 * pseudo + 0.7 * mle
    assert torch.allclose(loss2, expected2, atol=1e-6)
