# %%
from profiling_utils import create_remote_backend_from_triplets
from rag_feature_store import SentenceTransformerFeatureStore
from rag_graph_store import NeighborSamplingRAGGraphStore
from torch_geometric.loader import RAGQueryLoader
from torch_geometric.nn.nlp import SentenceTransformer
from torch_geometric.datasets.updated_web_qsp_dataset import preprocess_triplet, retrieval_via_pcst
from torch_geometric.data import get_features_for_triplets_groups, Data
from itertools import chain
import torch
from typing import Tuple
import tqdm
import pandas as pd


# %%
triplets = torch.load('wikimultihopqa_full_graph.pt')

# %%
df = pd.read_csv('wikimultihopqa_cleaned.csv')
questions = df['question_text'][:10]

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentenceTransformer(model_name='sentence-transformers/all-roberta-large-v1').to(device)

# %%
fs, gs = create_remote_backend_from_triplets(triplets=triplets, node_embedding_model=model, node_method_to_call="encode", path="backend", pre_transform=preprocess_triplet, node_method_kwargs={"batch_size": 256}, graph_db=NeighborSamplingRAGGraphStore, feature_db=SentenceTransformerFeatureStore).load()

# %%
query_loader = RAGQueryLoader(data=(fs, gs), seed_nodes_kwargs={"k_nodes": 10}, seed_edges_kwargs={"k_edges": 10}, sampler_kwargs={"num_neighbors": [40]*3}, local_filter=retrieval_via_pcst)

# %%
subgs = []
for subg in tqdm.tqdm((query_loader.query(q) for q in questions)):
    subgs.append(subg)

torch.save(subgs, 'subg_results.pt')
