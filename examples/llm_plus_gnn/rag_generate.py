# %%
from profiling_utils import create_remote_backend_from_triplets
from rag_feature_store import SentenceTransformerFeatureStore
from rag_graph_store import NeighborSamplingRAGGraphStore
from torch_geometric.loader import RAGQueryLoader
from torch_geometric.datasets import UpdatedWebQSPDataset
from torch_geometric.nn.nlp import SentenceTransformer
from torch_geometric.datasets.updated_web_qsp_dataset import preprocess_triplet, retrieval_via_pcst
from torch_geometric.data import get_features_for_triplets_groups, Data
from itertools import chain
import torch
from typing import Tuple
import tqdm
import pandas as pd

# %%
ds = UpdatedWebQSPDataset("small_dataset", force_reload=True, limit=20)

# %%
triplets = chain.from_iterable((d['graph'] for d in ds.raw_dataset))

# %%
questions = ds.raw_dataset['question']
questions

# %%
ground_truth_graphs = get_features_for_triplets_groups(ds.indexer, (d['graph'] for d in ds.raw_dataset), pre_transform=preprocess_triplet)
num_edges = len(ds.indexer._edges)

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentenceTransformer(model_name='sentence-transformers/all-roberta-large-v1').to(device)

# %%
fs, gs = create_remote_backend_from_triplets(triplets=triplets, node_embedding_model=model, node_method_to_call="encode", path="backend", pre_transform=preprocess_triplet, node_method_kwargs={"batch_size": 256}, graph_db=NeighborSamplingRAGGraphStore, feature_db=SentenceTransformerFeatureStore).load()

# %%
query_loader = RAGQueryLoader(data=(fs, gs), seed_nodes_kwargs={"k_nodes": 10}, seed_edges_kwargs={"k_edges": 10}, sampler_kwargs={"num_neighbors": [40]*3})

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
    tn = len(total_e-(subg_e | gt_e))
    return (tp+tn)/num_edges
def check_retrieval_precision(subg: Data, ground_truth: Data):
    subg_e, gt_e = _eidx_helper(subg, ground_truth)
    return len(subg_e & gt_e) / len(subg_e)
def check_retrieval_recall(subg: Data, ground_truth: Data):
    subg_e, gt_e = _eidx_helper(subg, ground_truth)
    return len(subg_e & gt_e) / len(gt_e)

# %%
retrieval_stats = {"precision": [], "recall": [], "accuracy": []}
subgs = []
for subg, gt in tqdm.tqdm(zip((query_loader.query(q) for q in questions), ground_truth_graphs)):
    retrieval_stats["precision"].append(check_retrieval_precision(subg, gt))
    retrieval_stats["recall"].append(check_retrieval_recall(subg, gt))
    retrieval_stats["accuracy"].append(check_retrieval_accuracy(subg, gt, num_edges))
    subgs.append(subg)

pd.DataFrame.from_dict(retrieval_stats).to_csv('metadata.csv')
torch.save(subgs, 'subg_results.pt')

