import argparse
import os
from itertools import chain

import datasets
import torch

from g_retriever import adjust_learning_rate, get_loss, inference_step
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from torch_geometric import seed_everything
from torch_geometric.loader import DataLoader, RAGQueryLoader
from torch_geometric.nn.models import GAT, GRetriever
from torch_geometric.nn.nlp import LLM, TXT2KG, SentenceTransformer
from torch_geometric.utils.rag.backend_utils import (
    create_remote_backend_from_triplets,
    make_pcst_filter,
    preprocess_triplet,
)
from torch_geometric.utils.rag.feature_store import (
    SentenceTransformerFeatureStore,
)
from torch_geometric.utils.rag.graph_store import NeighborSamplingRAGGraphStore

if __name__ == '__main__':
    seed_everything(50)
    parser = argparse.ArgumentParser()
    parser.add_argument('--NV_NIM_MODEL', type=str,
                        default="nvidia/llama-3.1-nemotron-70b-instruct",
                        help="The NIM LLM to use for TXT2KG")
    parser.add_argument('--NV_NIM_KEY', type=str, default="")
    parser.add_argument(
        '--chunk_size', type=int, default=512, help=
        "When splitting context documents, the maximum number of characters per chunk."
    )
    parser.add_argument('--gnn_hidden_channels', type=int, default=1024)
    parser.add_argument('--num_gnn_layers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--llm_generator_name', type=str,
                        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                        help="the LLM to use for Generation")
    args = parser.parse_args()
    lr = args.lr
    num_epochs = args.epochs
    # see examples/llm/hotpot_qa.py for example of using local LM
    kg_maker = TXT2KG(
        NVIDIA_NIM_MODEL=args.NV_NIM_MODEL,
        NVIDIA_API_KEY=args.NV_NIM_KEY,
        chunk_size=args.chunk_size,
    )
    # Data Prep
    if os.path.exists("tech_qa.pt"):
        print("Re-using Saved TechQA KG-RAG Dataset...")
        data_lists = torch.load("tech_qa.pt", weights_only=False)
    else:
        # Use training set for simplicity since our retrieval method is nonparametric
        rawset = datasets.load_dataset('rojagtap/tech-qa',
                                       trust_remote_code=True)
        data_lists = {"train": [], "validation": [], "test": []}
        # Build KG
        num_data_pts = len(rawset)
        for split_str in data_lists.keys():
            if split_str == "test":
                """
                Skip test since it is just a subset of val,
                so it's a waste of time to re-parse triples.
                """
                break
            i = 0
            for data_point in tqdm(
                    rawset[split_str],
                    desc="Extracting triples from " + str(split_str)):
                i += 1
                if i > 20:
                    break
                q = data_point["question"]
                a = data_point["answer"]
                context_doc = data_point["document"]

                QA_pair = (q, a)
                kg_maker.add_doc_2_KG(
                    txt=context_doc,
                    QA_pair=QA_pair,
                )
            kg_maker.save_kg("hotpot_kg.pt")
            relevant_triples = kg_maker.relevant_triples

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
            model_name='sentence-transformers/all-roberta-large-v1').to(device)
        fs, gs = create_remote_backend_from_triplets(
            triplets=triples, node_embedding_model=model,
            node_method_to_call="encode", path="backend",
            pre_transform=preprocess_triplet, node_method_kwargs={
                "batch_size": min(len(triples), 256)
            }, graph_db=NeighborSamplingRAGGraphStore,
            feature_db=SentenceTransformerFeatureStore).load()
        """
        NOTE: these retriever hyperparams are very important.
        Tuning may be needed for custom data...
        """

        # k for KNN
        knn_neighsample_bs = 1024
        # number of neighbors for each seed node selected by KNN
        fanout = 100
        # number of hops for neighborsampling
        num_hops = 2
        query_loader = RAGQueryLoader(
            data=(fs, gs), seed_nodes_kwargs={"k_nodes": knn_neighsample_bs},
            sampler_kwargs={"num_neighbors": [fanout] * num_hops},
            local_filter=make_pcst_filter(triples, model))

        for split_str in data_lists.keys():
            i = 0
            for data_point in tqdm(rawset[split_str], desc="Building dataset"):
                i += 1
                if i > 20:
                    break
                QA_pair = (data_point["question"], data_point["answer"])
                golden_triples = relevant_triples[QA_pair]

                q = QA_pair[0]
                subgraph = query_loader.query(q)
                subgraph.label = QA_pair[1]
                """
                store for golden triples for demo purpose:
                see how (GNN+LLM vs LLM) w/ (golden retrieval, normal retriever) for (GraphRAG)
                vs existing SOTA vector RAG from gilberto
                first comparison is for knowing how much retriever and model each matters
                """
                # Again, redundant since TXT2KG already provides lowercase
                # in case loading a KG that was made some other way without lowercase
                # golden_triples = [(i[0].lower(), i[1].lower(), i[2].lower())
                # for i in golden_triples]
                # subgraph.golden_triples = golden_triples
                data_lists[split_str].append(subgraph)
        torch.save(data_lists, "tech_qa.pt")
    ##### Done Prepping Data!

    # Training
    batch_size = args.batch_size
    eval_batch_size = args.eval_batch_size
    hidden_channels = args.gnn_hidden_channels
    num_gnn_layers = args.num_gnn_layers
    train_loader = DataLoader(data_lists["train"], batch_size=batch_size,
                              drop_last=True, pin_memory=True, shuffle=True)
    val_loader = DataLoader(data_lists["validation"],
                            batch_size=eval_batch_size, drop_last=False,
                            pin_memory=True, shuffle=False)
    test_loader = DataLoader(data_lists["test"], batch_size=eval_batch_size,
                             drop_last=False, pin_memory=True, shuffle=False)

    # Create GNN model
    gnn = GAT(
        in_channels=1024,
        hidden_channels=hidden_channels,
        out_channels=1024,
        num_layers=num_gnn_layers,
        heads=4,
    )
    # Create optimizer
    llm = LLM(model_name=args.llm_generator_name)
    model = GRetriever(llm=llm, gnn=gnn)
    params = [p for _, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {
            'params': params,
            'lr': lr,
            'weight_decay': 0.05
        },
    ], betas=(0.9, 0.95))

    # Initialize best epoch and best validation loss
    best_epoch = 0
    best_val_loss = float('inf')

    # Train model
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_str = f'Epoch: {epoch + 1}|{num_epochs}'
        loader = tqdm(train_loader, desc=epoch_str)
        for step, batch in enumerate(loader):
            optimizer.zero_grad()
            loss = get_loss(model, batch)
            loss.backward()

            clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1)

            if (step + 1) % 2 == 0:
                adjust_learning_rate(optimizer.param_groups[0], lr,
                                     step / len(train_loader) + epoch,
                                     num_epochs)

            optimizer.step()
            epoch_loss = epoch_loss + float(loss)

            if (step + 1) % 2 == 0:
                lr = optimizer.param_groups[0]['lr']
        train_loss = epoch_loss / len(train_loader)
        print(epoch_str + f', Train Loss: {train_loss:4f}')

        # Evaluate model
        val_loss = 0
        eval_output = []
        model.eval()
        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                loss = get_loss(model, batch)
                val_loss += loss.item()
            val_loss = val_loss / len(val_loader)
            print(epoch_str + f", Val Loss: {val_loss:4f}")
    # Clean up memory
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    model.eval()

    # Test
    metrics = []

    def eval(pred, answer):
        # add eval from gilberto
        return 1.0

    for test_batch in tqdm(test_loader, desc="Test:"):
        metrics.append(
            eval(inference_step(model, test_batch), test_batch.label))
    avg_metrics = sum(metrics) / len(metrics)
    print("Avg metric=", avg_metrics)
