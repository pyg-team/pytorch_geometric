import argparse
import os
from itertools import chain

import datasets
import torch
from tqdm import tqdm

from torch_geometric import seed_everything
from torch_geometric.datasets.web_qsp_dataset import preprocess_triplet
from torch_geometric.loader import RAGQueryLoader
from torch_geometric.nn.nlp import TXT2KG, SentenceTransformer
from torch_geometric.utils.rag.backend_utils import (
    create_remote_backend_from_triplets,
    make_pcst_filter,
)
from torch_geometric.utils.rag.feature_store import (
    SentenceTransformerFeatureStore,
)
from torch_geometric.utils.rag.graph_store import NeighborSamplingRAGGraphStore

if __name__ == '__main__':
    seed_everything(50)
    parser = argparse.ArgumentParser()
    parser.add_argument('--NV_NIM_KEY', type=str, default="")
    parser.add_argument('--local_lm', action="store_true")
    parser.add_argument('--percent_data', type=float, default=1.0)
    parser.add_argument('--chunk_size', type=int, default=512)
    args = parser.parse_args()
    assert args.percent_data <= 100 and args.percent_data > 0
    if args.local_lm:
        kg_maker = TXT2KG(
            local_LM=True,
            chunk_size=args.chunk_size,
        )
    else:
        kg_maker = TXT2KG(
            NVIDIA_API_KEY=args.NV_NIM_KEY,
            chunk_size=args.chunk_size,
        )
    if os.path.exists("hotpot_kg.pt"):
        print("Re-using existing KG...")
        relevant_triples = torch.load("hotpot_kg.pt")
    else:
        # Use training set for simplicity since our retrieval method is nonparametric
        raw_dataset = datasets.load_dataset('hotpotqa/hotpot_qa', 'fullwiki',
                                            trust_remote_code=True)["train"]
        # Build KG
        num_data_pts = len(raw_dataset)
        data_idxs = torch.randperm(num_data_pts)[0:int(num_data_pts *
                                                       args.percent_data /
                                                       100.0)]
        for idx in tqdm(data_idxs, desc="Building KG"):
            data_point = raw_dataset[int(idx)]
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
        kg_maker.save_kg("hotpot_kg.pt")
        relevant_triples = kg_maker.relevant_triples
        if args.local_lm:
            print("Total number of context characters parsed by LLM",
                  kg_maker.total_chars_parsed)
            print(
                "Average number of context characters parsed by LLM per second=",
                kg_maker.avg_chars_parsed_per_sec)

    print("Size of KG (number of triples) =",
          sum([len(rel_trips) for rel_trips in relevant_triples.values()]))

    triples = list(
        chain.from_iterable(triple_set
                            for triple_set in relevant_triples.values()))
    # debug
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(
        model_name='sentence-transformers/all-roberta-large-v1').to(device)
    fs, gs = create_remote_backend_from_triplets(
        triplets=triples, node_embedding_model=model,
        node_method_to_call="encode", path="backend",
        pre_transform=preprocess_triplet, node_method_kwargs={
            "batch_size": min(len(triples), 256)
        }, graph_db=NeighborSamplingRAGGraphStore,
        feature_db=SentenceTransformerFeatureStore).load()
    query_loader = RAGQueryLoader(
        data=(fs, gs), seed_nodes_kwargs={"k_nodes":
                                          5}, seed_edges_kwargs={"k_edges": 5},
        sampler_kwargs={"num_neighbors": [50] * 2},
        local_filter=make_pcst_filter(triples, model))
    """
    approx precision = num_relevant_out_of_retrieved/num_retrieved_triples
    We will use precision as a proxy for recall. This is because for recall,
    we must know how many relevant triples exist for each question,
    but this is not known.
    """
    precisions = []
    for QA_pair in relevant_triples.keys():
        golden_triples = relevant_triples[QA_pair]
        q = QA_pair[0]
        retrieved_subgraph = query_loader.query(q)
        retrieved_triples = retrieved_subgraph.triples
        print("Q=", q)
        print("A=", QA_pair[1])
        print("retrieved_triples =", retrieved_triples)
        num_relevant_out_of_retrieved = float(
            sum([
                int(bool(retrieved_triple in golden_triples))
                for retrieved_triple in retrieved_triples
            ]))
        precisions.append(num_relevant_out_of_retrieved /
                          len(retrieved_triples))
    approx_precision = sum(precisions) / len(precisions)
    print("approx_precision =", approx_precision)
