import argparse
import os
from itertools import chain

import datasets
import torch
from tqdm import tqdm

from torch_geometric import seed_everything
from torch_geometric.loader import RAGQueryLoader
from torch_geometric.nn.nlp import TXT2KG, SentenceTransformer
from torch_geometric.utils.rag.backend_utils import (
    create_remote_backend_from_triplets,
    make_pcst_filter,
    preprocess_triplet,
)
from torch_geometric.utils.rag.feature_store import ModernBertFeatureStore
from torch_geometric.utils.rag.graph_store import NeighborSamplingRAGGraphStore

if __name__ == '__main__':
    seed_everything(50)
    parser = argparse.ArgumentParser()
    parser.add_argument('--NV_NIM_MODEL', type=str,
                        default="nvidia/llama-3.1-nemotron-70b-instruct")
    parser.add_argument('--NV_NIM_KEY', type=str, default="")
    parser.add_argument('--local_lm', action="store_true")
    parser.add_argument('--percent_data', type=float, default=1.0)
    parser.add_argument(
        '--chunk_size', type=int, default=512,
        help="When splitting context documents for txt2kg,\
        the maximum number of characters per chunk.")
    parser.add_argument('--checkpointing', action="store_true")
    parser.add_argument('--verbose', action="store_true")
    args = parser.parse_args()
    assert args.percent_data <= 100 and args.percent_data > 0
    if args.local_lm:
        kg_maker = TXT2KG(
            local_LM=True,
            chunk_size=args.chunk_size,
        )
    else:
        kg_maker = TXT2KG(
            NVIDIA_NIM_MODEL=args.NV_NIM_MODEL,
            NVIDIA_API_KEY=args.NV_NIM_KEY,
            chunk_size=args.chunk_size,
        )
    if os.path.exists("hotpot_kg.pt"):
        print("Re-using existing KG...")
        relevant_triples = torch.load("hotpot_kg.pt", weights_only=False)
    elif os.path.exists("checkpoint_kg.pt"):
        print("Re-using existing checkpoint...")
        relevant_triples = torch.load("checkpoint_kg.pt", weights_only=False)
    else:
        # training set for simplicity since naive retrieval is nonparametric
        raw_dataset = datasets.load_dataset('hotpotqa/hotpot_qa', 'fullwiki',
                                            trust_remote_code=True)["train"]
        # Build KG
        num_data_pts = len(raw_dataset)
        data_idxs = torch.randperm(num_data_pts)[0:int(num_data_pts *
                                                       args.percent_data /
                                                       100.0)]
        if args.checkpointing:
            five_percent = int(len(data_idxs) / 20)
            count = 0
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
            if args.checkpointing:
                count += 1
                if count == five_percent:
                    print("checkpointing...")
                    count = 0
                    kg_maker.save_kg("checkpoint_kg.pt")
        kg_maker.save_kg("hotpot_kg.pt")
        if args.checkpointing:
            # delete checkpoint
            os.remove("checkpoint_kg.pt")

        relevant_triples = kg_maker.relevant_triples
        if args.local_lm:
            print("Total # of context characters parsed by LLM",
                  kg_maker.total_chars_parsed)
            print("Average # of context characters parsed by LLM per second=",
                  kg_maker.avg_chars_parsed_per_sec)

    triples = list(
        chain.from_iterable(triple_set
                            for triple_set in relevant_triples.values()))
    # redundant since TXT2KG already provides lowercase.
    # in case loading a KG that was made some other way without lowercase
    triples = [(i[0].lower(), i[1].lower(), i[2].lower()) for i in triples]
    # Make sure no duplicate triples for KG indexing
    triples = list(dict.fromkeys(triples))

    print("Size of KG (number of triples) =", len(triples))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(
        model_name='Alibaba-NLP/gte-modernbert-base').to(device)
    fs, gs = create_remote_backend_from_triplets(
        triplets=triples, node_embedding_model=model,
        node_method_to_call="encode", path="backend",
        pre_transform=preprocess_triplet, node_method_kwargs={
            "batch_size": min(len(triples), 256)
        }, graph_db=NeighborSamplingRAGGraphStore,
        feature_db=ModernBertFeatureStore).load()
    """
    NOTE: these retriever hyperparams are very important.
    Tuning may be needed for custom data...
    """
    # k for KNN
    knn_neighsample_bs = 4096
    # number of neighbors for each seed node selected by KNN
    fanout = 400
    # number of hops for neighborsampling
    num_hops = 2
    local_filter_kwargs = {
        "topk": 5,  # nodes
        "topk_e": 5,  # edges
        "cost_e": .5,  # edge cost
        "num_clusters": 10,  # num clusters
    }
    query_loader = RAGQueryLoader(
        data=(fs, gs), seed_nodes_kwargs={"k_nodes": knn_neighsample_bs},
        sampler_kwargs={"num_neighbors": [fanout] * num_hops},
        local_filter=make_pcst_filter(triples, model),
        local_filter_kwargs=local_filter_kwargs)
    """
    approx precision = num_golden_out_of_retrieved/num_retrieved_triples
    These are rough approximations since we do not know exactly which
    golden triples are actually relevant.
    """
    precisions = []
    if args.verbose:
        loader = relevant_triples.keys()
    else:
        loader = tqdm(relevant_triples.keys())
    for QA_pair in tqdm(relevant_triples.keys(), disable=args.verbose):
        golden_triples = relevant_triples[QA_pair]
        # Again, redundant since TXT2KG already provides lowercase
        # in case loading a KG that was made some other way without lowercase
        golden_triples = [(i[0].lower(), i[1].lower(), i[2].lower())
                          for i in golden_triples]
        q = QA_pair[0]
        retrieved_subgraph = query_loader.query(q)
        retrieved_triples = retrieved_subgraph.triples

        if args.verbose:
            print("Q=", q)
            print("A=", QA_pair[1])
            print("retrieved_triples =", retrieved_triples)

        num_relevant_out_of_retrieved = float(
            sum(retrieved_triple in golden_triples
                for retrieved_triple in retrieved_triples))
        precisions.append(num_relevant_out_of_retrieved /
                          max(len(retrieved_triples), 1))  # handle div by 0
    approx_precision = sum(precisions) / len(precisions)
    print("Min # of Extracted Triples =", min(extracted_triple_sizes))
    print("Max # of Extracted Triples =", max(extracted_triple_sizes))
    print("Average # of Extracted Triples =",
          sum(extracted_triple_sizes) / len(extracted_triple_sizes))
    print("approx_precision =",
          str(round(approx_precision * 100.0, 2)) + "% **")
    print("**:rough approximations since we do not know\
        exactly which 'golden' triples are actually relevant.")
