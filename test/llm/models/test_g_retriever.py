import gc

import torch

from torch_geometric.llm.models import LLM, GRetriever
from torch_geometric.nn import GAT
from torch_geometric.testing import onlyRAG, withPackage


@onlyRAG
@withPackage('transformers', 'sentencepiece', 'accelerate')
def test_g_retriever() -> None:
    llm = LLM(
        model_name='HuggingFaceTB/SmolLM-360M',
        num_params=1,
        dtype=torch.float16,
    )

    gnn = GAT(
        in_channels=1024,
        out_channels=1024,
        hidden_channels=1024,
        num_layers=2,
        heads=4,
        norm='batch_norm',
    )

    model = GRetriever(
        llm=llm,
        gnn=gnn,
    )
    assert str(model) == ('GRetriever(\n'
                          '  llm=LLM(HuggingFaceTB/SmolLM-360M),\n'
                          '  gnn=GAT(1024, 1024, num_layers=2),\n'
                          ')')

    x = torch.randn(10, 1024)
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
    ])
    edge_attr = torch.randn(edge_index.size(1), 1024)
    batch = torch.zeros(x.size(0), dtype=torch.long)

    question = ["Is PyG the best open-source GNN library?"]
    label = ["yes!"]

    # Test train:
    loss = model(question, x, edge_index, batch, label, edge_attr)
    assert loss >= 0

    # Test inference:
    pred = model.inference(question, x, edge_index, batch, edge_attr)
    assert len(pred) == 1
    del model
    gc.collect()
    torch.cuda.empty_cache()


@onlyRAG
@withPackage('transformers', 'sentencepiece', 'accelerate')
def test_g_retriever_many_tokens() -> None:
    llm = LLM(
        model_name='HuggingFaceTB/SmolLM-360M',
        num_params=1,
        dtype=torch.float16,
    )

    gnn = GAT(
        in_channels=1024,
        out_channels=1024,
        hidden_channels=1024,
        num_layers=2,
        heads=4,
        norm='batch_norm',
    )

    model = GRetriever(
        llm=llm,
        gnn=gnn,
        mlp_out_tokens=2,
    )
    assert str(model) == ('GRetriever(\n'
                          '  llm=LLM(HuggingFaceTB/SmolLM-360M),\n'
                          '  gnn=GAT(1024, 1024, num_layers=2),\n'
                          ')')

    x = torch.randn(10, 1024)
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
    ])
    edge_attr = torch.randn(edge_index.size(1), 1024)
    batch = torch.zeros(x.size(0), dtype=torch.long)

    question = ["Is PyG the best open-source GNN library?"]
    label = ["yes!"]

    # Test train:
    loss = model(question, x, edge_index, batch, label, edge_attr)
    assert loss >= 0

    # Test inference:
    pred = model.inference(question, x, edge_index, batch, edge_attr)
    assert len(pred) == 1
    del model
    gc.collect()
    torch.cuda.empty_cache()
