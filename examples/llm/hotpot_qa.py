import argparse

import datasets
from tqdm import tqdm

from torch_geometric.nn.nlp import TXT2KG

parser = argparse.ArgumentParser()
parser.add_argument('--NV_NIM_KEY', type=str, required=True)
parser.add_argument('--percent_data', type=int, default=10)
args = parser.parse_args()
assert args.percent_data <= 100 and args.percent_data > 0
kg_maker = TXT2KG(
    NVIDIA_API_KEY=args.NV_NIM_KEY,
    chunk_size=512,
)

# Use training set for simplicity since our retrieval method is nonparametric
raw_dataset = datasets.load_dataset('hotpotqa/hotpot_qa', 'fullwiki')["train"]
# Build KG
num_data_pts = len(raw_dataset)
data_idxs = torch.randperm(num_data_pts)[0:int(num_data_pts *
                                               float(args.percent_data) /
                                               100.0)]
for idx in tqdm(data_idxs, desc="Building KG"):
    data_point = raw_dataset[idx]
    q = data_point["question"]
    a = data_point["answer"]
    context_doc = ''
    for i in data_point["context"]["sentences"]:
        for sentence in i:
            context_doc += sentence

    QA_pair = (q, a)
    kg_maker.add_doc_2_KG(
        txt=context_doc,
        QA_pair=QA_pair,
    )
# (TODO) need rebase onto Zack's PR to be able to use the RAGQueryLoader
# Note: code below here will not work until the rebase is done
from itertools import chain

from g_retriever_utils import apply_retrieval_via_pcst
from rag_backend_utils import create_remote_backend_from_triplets
from rag_feature_store import SentenceTransformerFeatureStore
from rag_graph_store import NeighborSamplingRAGGraphStore

from torch_geometric.datasets.web_qsp_dataset import (
    preprocess_triplet,
    retrieval_via_pcst,
)
from torch_geometric.loader import RAGQueryLoader
from torch_geometric.nn.nlp import SentenceTransformer

triples = chain.from_iterable(triple_set for triple_set in kg_maker.relevant_triples.values())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentenceTransformer(
    model_name='sentence-transformers/all-roberta-large-v1').to(device)
fs, gs = create_remote_backend_from_triplets(
    triplets=triplets, node_embedding_model=model,
    node_method_to_call="encode", path="backend",
    pre_transform=preprocess_triplet, node_method_kwargs={
        "batch_size": 256
    }, graph_db=NeighborSamplingRAGGraphStore,
    feature_db=SentenceTransformerFeatureStore).load()

query_loader = RAGQueryLoader(data=(fs, gs), seed_nodes_kwargs={"k_nodes": 5},
                              seed_edges_kwargs={"k_edges": 5},
                              sampler_kwargs={"num_neighbors": [50] * 2},
                              local_filter=transform)
"""
approx precision = num_relevant_out_of_retrieved/num_retrieved_triples
We will use precision as a proxy for recall. This is because for recall,
we must know how many relevant triples exist for each question,
but this is not known.
"""
precisions = []
for QA_pair in kg_maker.relevant_triples.keys():
    relevant_triples = kg_maker.relevant_triples[QA_pair]
    q = QA_pair[0]
    retrieved_subgraph = query_loader.query(q)
    print("retrieved_subgraph=", retrieved_subgraph)
    retrieved_triples = # extract triples from subgraph
    num_relevant_out_of_retrieved = float(sum([int(bool(retrieved_triple in relevant_triples)) for retrieved_triple in retrieved_triples]))
    precisions.append(num_relevant_out_of_retrieved/len(retrieved_triples))
approx_precision = mean(precisions)
