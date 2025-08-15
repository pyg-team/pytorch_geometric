# %%
import argparse
import sys
from itertools import chain
from typing import Tuple

import torch

from torch_geometric.data import Data, get_features_for_triplets_groups
from torch_geometric.datasets import WebQSPDataset
from torch_geometric.datasets.web_qsp_dataset import (
    preprocess_triplet,
    retrieval_via_pcst,
)
from torch_geometric.loader import rag_loader
from torch_geometric.nn.nlp import SentenceTransformer
from torch_geometric.profile.nvtx import nvtxit

sys.path.append('..')
from g_retriever_utils.rag_backend_utils import \
    create_remote_backend_from_triplets  # noqa: E402
from g_retriever_utils.rag_feature_store import \
    SentenceTransformerFeatureStore  # noqa: E402
from g_retriever_utils.rag_graph_store import \
    NeighborSamplingRAGGraphStore  # noqa: E402

# %%
# Patch FeatureStore and GraphStore

SentenceTransformerFeatureStore.retrieve_seed_nodes = nvtxit()(
    SentenceTransformerFeatureStore.retrieve_seed_nodes)
SentenceTransformerFeatureStore.retrieve_seed_edges = nvtxit()(
    SentenceTransformerFeatureStore.retrieve_seed_edges)
SentenceTransformerFeatureStore.load_subgraph = nvtxit()(
    SentenceTransformerFeatureStore.load_subgraph)
NeighborSamplingRAGGraphStore.sample_subgraph = nvtxit()(
    NeighborSamplingRAGGraphStore.sample_subgraph)
rag_loader.RAGQueryLoader.query = nvtxit()(rag_loader.RAGQueryLoader.query)

# %%
ds = WebQSPDataset("small_ds_1", force_reload=True, limit=10)

# %%
triplets = list(chain.from_iterable(d['graph'] for d in ds.raw_dataset))

# %%
questions = ds.raw_dataset['question']

# %%
ground_truth_graphs = get_features_for_triplets_groups(
    ds.indexer, (d['graph'] for d in ds.raw_dataset),
    pre_transform=preprocess_triplet)
num_edges = len(ds.indexer._edges)

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentenceTransformer('sentence-transformers/all-roberta-large-v1').to(
    device)

# %%
fs, gs = create_remote_backend_from_triplets(
    triplets=triplets, node_embedding_model=model,
    node_method_to_call="encode", path="backend",
    pre_transform=preprocess_triplet, node_method_kwargs={
        "batch_size": 256
    }, graph_db=NeighborSamplingRAGGraphStore,
    feature_db=SentenceTransformerFeatureStore).load()

# %%


@nvtxit()
def apply_retrieval_via_pcst(graph: Data, query: str, topk: int = 3,
                             topk_e: int = 3,
                             cost_e: float = 0.5) -> Tuple[Data, str]:
    q_emb = model.encode(query)
    textual_nodes = ds.textual_nodes.iloc[graph["node_idx"]].reset_index()
    textual_edges = ds.textual_edges.iloc[graph["edge_idx"]].reset_index()
    out_graph, desc = retrieval_via_pcst(graph, q_emb, textual_nodes,
                                         textual_edges, topk, topk_e, cost_e)
    out_graph["desc"] = desc
    return graph


# %%
query_loader = rag_loader.RAGQueryLoader(
    data=(fs, gs), seed_nodes_kwargs={"k_nodes":
                                      10}, seed_edges_kwargs={"k_edges": 10},
    sampler_kwargs={"num_neighbors":
                    [40] * 10}, local_filter=apply_retrieval_via_pcst)


# %%
# Accuracy Metrics to be added to Profiler
def _eidx_helper(subg: Data, ground_truth: Data):
    subg_eidx, gt_eidx = subg.edge_idx, ground_truth.edge_idx
    if isinstance(subg_eidx, torch.Tensor):
        subg_eidx = subg_eidx.tolist()
    if isinstance(gt_eidx, torch.Tensor):
        gt_eidx = gt_eidx.tolist()
    subg_e = set(subg_eidx)
    gt_e = set(gt_eidx)
    return subg_e, gt_e


def check_retrieval_accuracy(subg: Data, ground_truth: Data, num_edges: int):
    subg_e, gt_e = _eidx_helper(subg, ground_truth)
    total_e = set(range(num_edges))
    tp = len(subg_e & gt_e)
    tn = len(total_e - (subg_e | gt_e))
    return (tp + tn) / num_edges


def check_retrieval_precision(subg: Data, ground_truth: Data):
    subg_e, gt_e = _eidx_helper(subg, ground_truth)
    return len(subg_e & gt_e) / len(subg_e)


def check_retrieval_recall(subg: Data, ground_truth: Data):
    subg_e, gt_e = _eidx_helper(subg, ground_truth)
    return len(subg_e & gt_e) / len(gt_e)


# %%


@nvtxit()
def _run_eval():
    for subg, gt in zip((query_loader.query(q) for q in questions),
                        ground_truth_graphs):
        print(check_retrieval_accuracy(subg, gt, num_edges),
              check_retrieval_precision(subg, gt),
              check_retrieval_recall(subg, gt))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-torch-kernels", "-k", action="store_true")
    args = parser.parse_args()
    if args.capture_torch_kernels:
        with torch.autograd.profiler.emit_nvtx():
            _run_eval()
    else:
        _run_eval()
