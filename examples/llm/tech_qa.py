import argparse
import os
from itertools import chain

import torch
from datasets import load_dataset
from g_retriever import (
    adjust_learning_rate,
    get_loss,
    inference_step,
    load_params_dict,
    save_params_dict,
)
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from torch_geometric import seed_everything
from torch_geometric.loader import DataLoader, RAGQueryLoader
from torch_geometric.nn import (
    GAT,
    LLM,
    TXT2KG,
    GRetriever,
    LLMJudge,
    SentenceTransformer,
)
from torch_geometric.utils.rag.backend_utils import (
    create_remote_backend_from_triplets,
    make_pcst_filter,
    preprocess_triplet,
)
from torch_geometric.utils.rag.feature_store import (
    SentenceTransformerFeatureStore,
)
from torch_geometric.utils.rag.graph_store import NeighborSamplingRAGGraphStore

# Define constants for better readability
NV_NIM_MODEL_DEFAULT = "nvidia/llama-3.1-nemotron-70b-instruct"
LLM_GENERATOR_NAME_DEFAULT = "meta-llama/Meta-Llama-3.1-8B-Instruct"
CHUNK_SIZE_DEFAULT = 512
GNN_HID_CHANNELS_DEFAULT = 1024
GNN_LAYERS_DEFAULT = 4
LR_DEFAULT = 1e-5
EPOCHS_DEFAULT = 2
BATCH_SIZE_DEFAULT = 8
EVAL_BATCH_SIZE_DEFAULT = 16


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--NV_NIM_MODEL', type=str,
                        default=NV_NIM_MODEL_DEFAULT,
                        help="The NIM LLM to use for TXT2KG")
    parser.add_argument('--NV_NIM_KEY', type=str, default="",
                        help="NVIDIA API key")
    parser.add_argument(
        '--chunk_size', type=int, default=512,
        help="When splitting context documents for txt2kg,\
        the maximum number of characters per chunk.")
    parser.add_argument('--gnn_hidden_channels', type=int,
                        default=GNN_HID_CHANNELS_DEFAULT,
                        help="Hidden channels for GNN")
    parser.add_argument('--num_gnn_layers', type=int,
                        default=GNN_LAYERS_DEFAULT,
                        help="Number of GNN layers")
    parser.add_argument('--lr', type=float, default=LR_DEFAULT,
                        help="Learning rate")
    parser.add_argument('--epochs', type=int, default=EPOCHS_DEFAULT,
                        help="Number of epochs")
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help="Batch size")
    parser.add_argument('--eval_batch_size', type=int,
                        default=EVAL_BATCH_SIZE_DEFAULT,
                        help="Evaluation batch size")
    parser.add_argument('--llm_generator_name', type=str,
                        default=LLM_GENERATOR_NAME_DEFAULT,
                        help="The LLM to use for Generation")
    return parser.parse_args()


def make_dataset(args):
    if os.path.exists("tech_qa.pt"):
        print("Re-using Saved TechQA KG-RAG Dataset...")
        return torch.load("tech_qa.pt", weights_only=False)
    else:
        if os.path.exists("tech_qa_just_triples.pt"):
            torch.load("tech_qa.pt", weights_only=False)
        else:
            rawset = load_dataset('rojagtap/tech-qa', trust_remote_code=True)
            data_lists = {"train": [], "validation": [], "test": []}
            kg_maker = TXT2KG(NVIDIA_NIM_MODEL=args.NV_NIM_MODEL,
                              NVIDIA_API_KEY=args.NV_NIM_KEY,
                              chunk_size=args.chunk_size)
            triples = []
            for split_str in data_lists.keys():
                if split_str == "test":
                    """
                    Skip test since it is just a subset of val,
                    so it's a waste of time to re-parse triples.
                    """
                    break
                for data_point in tqdm(
                        rawset[split_str],
                        desc="Extracting KG triples from " + str(split_str)):
                    q = data_point["question"]
                    a = data_point["answer"]
                    context_doc = data_point["document"]
                    QA_pair = (q, a)
                    kg_maker.add_doc_2_KG(txt=context_doc, QA_pair=QA_pair)
                kg_maker.save_kg("hotpot_kg.pt")
                relevant_triples = kg_maker.relevant_triples
                triples.extend(
                    list(
                        chain.from_iterable(
                            triple_set
                            for triple_set in relevant_triples.values())))
            triples = list(dict.fromkeys(triples))
            torch.save(data_lists, "tech_qa_just_triples.pt")
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
        local_filter_kwargs = {
            "topk": 5,  # nodes
            "topk_e": 5,  # edges
            "cost_e": .5  # edge cost
        }
        query_loader = RAGQueryLoader(
            data=(fs, gs), seed_nodes_kwargs={"k_nodes": knn_neighsample_bs},
            sampler_kwargs={"num_neighbors": [fanout] * num_hops},
            local_filter=make_pcst_filter(triples, model),
            local_filter_kwargs=local_filter_kwargs)
        for split_str in data_lists.keys():
            for data_point in tqdm(
                    rawset[split_str],
                    desc="Building " + str(split_str) + " dataset"):
                QA_pair = (data_point["question"], data_point["answer"])
                q = QA_pair[0]
                subgraph = query_loader.query(q)
                subgraph.label = QA_pair[1]
                data_lists[split_str].append(subgraph)
        torch.save(data_lists, "tech_qa.pt")
        return data_lists


def train(args, data_lists):
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
    gnn = GAT(in_channels=1024, hidden_channels=hidden_channels,
              out_channels=1024, num_layers=num_gnn_layers, heads=4)
    llm = LLM(model_name=args.llm_generator_name)
    model = GRetriever(llm=llm, gnn=gnn)
    save_name = "tech-qa-model.pt"
    if os.path.exists(save_name):
        print("Re-using saved G-retriever model for testing...")
        model = load_params_dict(model, save_name)
    else:
        params = [p for _, p in model.named_parameters() if p.requires_grad]
        lr = args.lr
        optimizer = torch.optim.AdamW([{
            'params': params,
            'lr': lr,
            'weight_decay': 0.05
        }], betas=(0.9, 0.95))
        float('inf')
        for epoch in range(args.epochs):
            model.train()
            epoch_loss = 0
            epoch_str = f'Epoch: {epoch + 1}|{args.epochs}'
            loader = tqdm(train_loader, desc=epoch_str)
            for step, batch in enumerate(loader):
                optimizer.zero_grad()
                loss = get_loss(model, batch)
                loss.backward()
                clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1)
                if (step + 1) % 2 == 0:
                    adjust_learning_rate(optimizer.param_groups[0], lr,
                                         step / len(train_loader) + epoch,
                                         args.epochs)
                optimizer.step()
                epoch_loss += float(loss)
                if (step + 1) % 2 == 0:
                    lr = optimizer.param_groups[0]['lr']
            train_loss = epoch_loss / len(train_loader)
            print(epoch_str + f', Train Loss: {train_loss:4f}')
            val_loss = 0
            model.eval()
            with torch.no_grad():
                for step, batch in enumerate(val_loader):
                    loss = get_loss(model, batch)
                    val_loss += loss.item()
                val_loss = val_loss / len(val_loader)
                print(epoch_str + f", Val Loss: {val_loss:4f}")
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        model.eval()
        save_params_dict(model, save_path=save_name)
    return model, test_loader


def test(model, test_loader, args):
    llm_judge = LLMJudge(args.NV_NIM_MODEL, args.NV_NIM_KEY)

    def eval(question: str, pred: str, correct_answer: str):
        # calculate the score based on pred and correct answer
        return llm_judge.score(question, pred, correct_answer)

    scores = []
    eval_tuples = []
    for test_batch in tqdm(test_loader, desc="Testing"):
        preds = (inference_step(model, test_batch))
        for question, pred, label in zip(test_batch.question, preds,
                                         test_batch.label):
            eval_tuples.append((question, pred, label))
    for question, pred, label in tqdm(eval_tuples, desc="Eval"):
        scores.append(eval(question, pred, label))
    avg_scores = sum(scores) / len(scores)
    print("Avg marlin accuracy=", avg_scores)


if __name__ == '__main__':
    # for reproducibility
    seed_everything(50)

    args = parse_args()
    data_lists = make_dataset(args)
    model, test_loader = train(args, data_lists)
    test(model, test_loader, args)
