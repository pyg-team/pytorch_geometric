# %%
import argparse
import sys
from typing import Tuple

import pandas as pd
import torch
import tqdm

from torch_geometric.data import Data
from torch_geometric.datasets.web_qsp_dataset import (
    preprocess_triplet,
    retrieval_via_pcst,
)
from torch_geometric.loader import RAGQueryLoader
from torch_geometric.nn.nlp import SentenceTransformer

sys.path.append('..')

from g_retriever_utils.rag_backend_utils import \
    create_remote_backend_from_triplets  # noqa: E402
from g_retriever_utils.rag_feature_store import \
    SentenceTransformerApproxFeatureStore  # noqa: E402
from g_retriever_utils.rag_graph_store import \
    NeighborSamplingRAGGraphStore  # noqa: E402

# %%
parser = argparse.ArgumentParser(
    description="Generate new multihop dataset for rag")
# TODO: Add more arguments for configuring rag params
parser.add_argument("--num_samples", type=int)
args = parser.parse_args()

# %%
triplets = torch.load('wikimultihopqa_full_graph.pt')

# %%
df = pd.read_csv('wikimultihopqa_cleaned.csv')
questions = df['question'][:args.num_samples]
labels = df['answer'][:args.num_samples]

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
    feature_db=SentenceTransformerApproxFeatureStore).load()

# %%

all_textual_nodes = pd.read_csv('wikimultihopqa_textual_nodes.csv')
all_textual_edges = pd.read_csv('wikimultihopqa_textual_edges.csv')


def apply_retrieval_via_pcst(graph: Data, query: str, topk: int = 3,
                             topk_e: int = 3,
                             cost_e: float = 0.5) -> Tuple[Data, str]:
    q_emb = model.encode(query)
    textual_nodes = all_textual_nodes.iloc[graph["node_idx"]].reset_index()
    textual_edges = all_textual_edges.iloc[graph["edge_idx"]].reset_index()
    out_graph, desc = retrieval_via_pcst(graph, q_emb, textual_nodes,
                                         textual_edges, topk, topk_e, cost_e)
    out_graph["desc"] = desc
    return out_graph


# %%
query_loader = RAGQueryLoader(data=(fs, gs), seed_nodes_kwargs={"k_nodes": 10},
                              seed_edges_kwargs={"k_edges": 10},
                              sampler_kwargs={"num_neighbors": [40] * 3},
                              local_filter=apply_retrieval_via_pcst)

# %%
subgs = []
for q, l in tqdm.tqdm(zip(questions, labels)):
    subg = query_loader.query(q)
    subg['question'] = q
    subg['label'] = l
    subgs.append(subg)

torch.save(subgs, 'subg_results.pt')
