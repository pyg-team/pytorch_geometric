# %%
import argparse
from itertools import chain
from typing import Tuple

import pandas as pd
import torch
import tqdm
from rag_backend_utils import create_remote_backend_from_triplets
from rag_feature_store import SentenceTransformerFeatureStore
from rag_graph_store import NeighborSamplingRAGGraphStore

from torch_geometric.data import Data
from torch_geometric.datasets import WebQSPDataset
from torch_geometric.datasets.web_qsp_dataset import (
    preprocess_triplet,
    retrieval_via_pcst,
)
from torch_geometric.loader import RAGQueryLoader
from torch_geometric.nn.nlp import SentenceTransformer

# %%
parser = argparse.ArgumentParser(
    description="""Generate new WebQSP subgraphs\n""" +
    """NOTE: Evaluating with smaller samples may result in""" +
    """ poorer performance for the trained models compared""" +
    """ to untrained models.""")
# TODO: Add more arguments for configuring rag params
parser.add_argument("--use_pcst", action="store_true")
parser.add_argument("--num_samples", type=int, default=4700)
parser.add_argument("--out_file", default="subg_results.pt")
args = parser.parse_args()

# %%
ds = WebQSPDataset("dataset", limit=args.num_samples, verbose=True,
                   force_reload=True)

# %%
triplets = chain.from_iterable(d['graph'] for d in ds.raw_dataset)

# %%
questions = ds.raw_dataset['question']

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentenceTransformer(
    model_name='sentence-transformers/all-roberta-large-v1').to(device)

# %%
fs, gs = create_remote_backend_from_triplets(
    triplets=triplets, node_embedding_model=model,
    node_method_to_call="encode", path="backend",
    pre_transform=preprocess_triplet, node_method_kwargs={
        "batch_size": 256
    }, graph_db=NeighborSamplingRAGGraphStore,
    feature_db=SentenceTransformerFeatureStore).load()

# %%


def apply_retrieval_via_pcst(graph: Data, query: str, topk: int = 3,
                             topk_e: int = 3,
                             cost_e: float = 0.5) -> Tuple[Data, str]:
    q_emb = model.encode(query)
    textual_nodes = ds.textual_nodes.iloc[graph["node_idx"]].reset_index()
    textual_edges = ds.textual_edges.iloc[graph["edge_idx"]].reset_index()
    out_graph, desc = retrieval_via_pcst(graph, q_emb, textual_nodes,
                                         textual_edges, topk, topk_e, cost_e)
    out_graph["desc"] = desc
    return out_graph


def apply_retrieval_with_text(graph: Data, query: str) -> Tuple[Data, str]:
    textual_nodes = ds.textual_nodes.iloc[graph["node_idx"]].reset_index()
    textual_edges = ds.textual_edges.iloc[graph["edge_idx"]].reset_index()
    desc = (
        textual_nodes.to_csv(index=False) + "\n" +
        textual_edges.to_csv(index=False, columns=["src", "edge_attr", "dst"]))
    graph["desc"] = desc
    return graph


transform = apply_retrieval_via_pcst \
    if args.use_pcst else apply_retrieval_with_text

query_loader = RAGQueryLoader(data=(fs, gs), seed_nodes_kwargs={"k_nodes": 5},
                              seed_edges_kwargs={"k_edges": 5},
                              sampler_kwargs={"num_neighbors": [50] * 2},
                              local_filter=transform)


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
retrieval_stats = {"precision": [], "recall": [], "accuracy": []}
subgs = []
node_len = []
edge_len = []
for subg in tqdm.tqdm(query_loader.query(q) for q in questions):
    subgs.append(subg)
    node_len.append(subg['x'].shape[0])
    edge_len.append(subg['edge_attr'].shape[0])

for i, subg in enumerate(subgs):
    subg['question'] = questions[i]
    subg['label'] = ds[i]['label']

pd.DataFrame.from_dict(retrieval_stats).to_csv(
    args.out_file.split('.')[0] + '_metadata.csv')
torch.save(subgs, args.out_file)
