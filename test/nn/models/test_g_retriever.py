import torch

from torch_geometric.data import Data
from torch_geometric.nn.models import GRetriever
from torch_geometric.testing import onlyFullTest, withPackage


def _setup_batch() -> Data:
    batch = Data()
    batch.question = ["Is PyG the best open-source GNN library?"]
    batch.label = ["yes!"]
    batch.num_nodes = 10
    batch.n_id = torch.arange(10).to(torch.int64)
    batch.num_edges = 20
    batch.x = torch.randn((10, 1024))
    batch.edge_attr = torch.randn((20, 1024))
    # Model expects batches sampled from Dataloader
    # hardcoding values for single item batch
    batch.batch = torch.zeros(batch.num_nodes).to(torch.int64)
    batch.ptr = torch.tensor([0, batch.num_nodes]).to(torch.int64)
    # simple cycle graph
    batch.edge_index = torch.tensor(
        [list(range(10)), list(range(1, 10)) + [0]])
    return batch


def _eval_train_call(model, batch):
    # test train
    loss = model(batch.question, batch.x, batch.edge_index, batch.batch,
                 batch.ptr, batch.label, batch.edge_attr)
    assert loss is not None


def _eval_inference_call(model, batch):
    # test inference
    pred = model.inference(batch.question, batch.x, batch.edge_index,
                           batch.batch, batch.ptr, batch.edge_attr)
    assert pred is not None


@onlyFullTest
@withPackage('transformers', 'sentencepiece', 'accelerate')
def test_g_retriever() -> None:
    batch = _setup_batch()

    model = GRetriever(
        llm_to_use="TinyLlama/TinyLlama-1.1B-Chat-v0.1",
        num_llm_params=1,  # 1 Billion
        mlp_out_dim=2048,
    )
    batch = batch.to(model.llm_device)

    _eval_train_call(model, batch)
    _eval_inference_call(model, batch)


@onlyFullTest
@withPackage('transformers', 'sentencepiece', 'accelerate')
def test_g_retriever_many_tokens() -> None:
    batch = _setup_batch()

    model = GRetriever(
        llm_to_use="TinyLlama/TinyLlama-1.1B-Chat-v0.1",
        num_llm_params=1,  # 1 Billion
        mlp_out_dim=2048,
        mlp_out_tokens=2)
    batch = batch.to(model.llm_device)

    _eval_train_call(model, batch)
    _eval_inference_call(model, batch)
