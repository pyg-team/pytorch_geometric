import gc

import pytest
import torch

from torch import nn
from torch_geometric.llm.models import LLM, GRetriever
from torch_geometric.nn import GAT
from torch_geometric.testing import onlyRAG, withPackage


@onlyRAG
@withPackage('transformers', 'sentencepiece', 'accelerate')
def test_g_retriever() -> None:
    llm = LLM(model_name='Qwen/Qwen3-0.6B', dtype=torch.float32,
              sys_prompt="You're an agent, answer my questions.")

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
                          '  llm=LLM(Qwen/Qwen3-0.6B),\n'
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
    del model, llm, gnn
    gc.collect()
    torch.cuda.empty_cache()


@onlyRAG
@withPackage('transformers', 'sentencepiece', 'accelerate')
def test_g_retriever_many_tokens() -> None:
    llm = LLM(model_name='Qwen/Qwen3-0.6B', dtype=torch.float32,
              sys_prompt="You're an agent, answer my questions.")

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
                          '  llm=LLM(Qwen/Qwen3-0.6B),\n'
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
    del model, llm, gnn
    gc.collect()
    torch.cuda.empty_cache()


class DummyHFModel(nn.Module):
    def __init__(self, hidden_size=8, vocab_size=100):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, hidden_size)

    def forward(self, inputs_embeds=None, **kwargs):
        class Output:
            def __init__(self, logits):
                self.logits = logits

        B, T, D = inputs_embeds.shape
        logits = torch.randn(B, T, vocab_size := 10, device=inputs_embeds.device)
        return Output(logits)


class DummyTokenizer:
    def __call__(self, texts, return_tensors=None, padding=True):
        if isinstance(texts, str):
            texts = [texts]

        batch = len(texts)
        seq_len = 4  # fixed dummy length

        return {
            "input_ids": torch.zeros(batch, seq_len, dtype=torch.long),
            "attention_mask": torch.ones(batch, seq_len, dtype=torch.long),
        }


class DummyLLM(nn.Module):
    def __init__(self, hidden_dim, vocab_size=100):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.word_embedding = nn.Embedding(vocab_size, hidden_dim)

        self.tokenizer = DummyTokenizer()
        self.device = torch.device("cpu")

    # required by some wrappers
    def get_input_embeddings(self):
        return self.word_embedding

    # used by GRetriever for prefix injection
    def _get_embeds(self, input_ids=None, inputs_embeds=None, **kwargs):
        if inputs_embeds is not None:
            return inputs_embeds
        return self.word_embedding(input_ids)

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        labels=None,
        **kwargs,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.word_embedding(input_ids)

        batch, seq, hidden = inputs_embeds.shape

        logits = torch.randn(batch, seq, hidden, device=inputs_embeds.device)

        return {
            "loss": torch.tensor(0.0, device=inputs_embeds.device),
            "logits": logits,
        }


class DummyGNN(nn.Module):
    """Simple GNN stub returning node embeddings."""
    def __init__(self, in_channels=4, out_channels=8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin = nn.Linear(in_channels, out_channels)


    def forward(self, *args, **kwargs):
        x = args[0]
        return self.lin(x)


@pytest.mark.parametrize("batch_size", [1, 3])
def test_gretriever_prefix_embedding_injection(batch_size):
    hidden_dim = 8
    num_nodes = 5

    llm = DummyLLM(hidden_dim)
    gnn = DummyGNN(in_channels=4, out_channels=8)

    model = GRetriever(
        llm=llm,
        gnn=gnn,
        mlp_out_tokens=2,
    )

    # graph inputs
    x = torch.randn(num_nodes, 4)
    edge_index = torch.tensor([[0,1,2],[1,2,3]])
    batch = torch.zeros(num_nodes, dtype=torch.long)

    # token ids
    questions = ["What is this graph?"] * batch_size
    labels = ["dummy answer"] * batch_size

    out = model(
    	x=x,
    	edge_index=edge_index,
    	batch=batch,
    	question=questions,
    	label=labels,
    )

    # basic correctness assertions
    assert hasattr(out, "logits")
    assert out.logits.shape[0] == batch_size
