# default args -> 40% test acc.
# 5-8% diff vs VectorRAG baselines
import argparse
import gc
import json
import os
import random
import re
import sys
from datetime import datetime
from glob import glob
from itertools import chain

try:
    import wandb
    wandb_available = True
except ImportError:
    wandb_available = False

import torch
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
from torch_geometric.nn.nlp.txt2kg import _chunk_text
from torch_geometric.utils.rag.backend_utils import (
    create_remote_backend_from_triplets,
    make_pcst_filter,
    preprocess_triplet,
)
from torch_geometric.utils.rag.feature_store import ModernBertFeatureStore
from torch_geometric.utils.rag.graph_store import NeighborSamplingRAGGraphStore

# Define constants for better readability
NV_NIM_MODEL_DEFAULT = "nvidia/llama-3.1-nemotron-70b-instruct"
LLM_GENERATOR_NAME_DEFAULT = "meta-llama/Meta-Llama-3.1-8B-Instruct"
CHUNK_SIZE_DEFAULT = 512
GNN_HID_CHANNELS_DEFAULT = 1024
GNN_LAYERS_DEFAULT = 4
LR_DEFAULT = 1e-5
EPOCHS_DEFAULT = 2
BATCH_SIZE_DEFAULT = 1
EVAL_BATCH_SIZE_DEFAULT = 2
LLM_GEN_MODE_DEFAULT = "full"
DEFAULT_ENDPOINT_URL = "https://integrate.api.nvidia.com/v1"
DOC_CHUNK_SIZE_DEFAULT = 8192


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--NV_NIM_MODEL', type=str,
                        default=NV_NIM_MODEL_DEFAULT,
                        help="The NIM LLM to use for TXT2KG for LLMJudge")
    parser.add_argument('--NV_NIM_KEY', type=str, default="",
                        help="NVIDIA API key")
    parser.add_argument(
        '--ENDPOINT_URL', type=str, default=DEFAULT_ENDPOINT_URL, help=
        "The URL hosting your model, in case you are not using the public NIM."
    )
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
    parser.add_argument('--doc_chunk_size', type=int,
                        default=DOC_CHUNK_SIZE_DEFAULT,
                        help="The chunk size to use for the LLM")
    parser.add_argument(
        '--llm_generator_mode', type=str, default=LLM_GEN_MODE_DEFAULT,
        choices=["frozen", "lora",
                 "full"], help="Whether to freeze the Generator LLM,\
                        use LORA, or fully finetune")
    parser.add_argument('--dont_save_model', action="store_true",
                        help="Whether to skip model saving.")
    parser.add_argument('--k_for_docs', type=int, default=2,
                        help="Number of docs to retrieve for each question.")
    parser.add_argument('--log_steps', type=int, default=30,
                        help="Log to wandb every N steps")
    parser.add_argument('--wandb_project', type=str, default="tech-qa",
                        help="Weights & Biases project name")
    parser.add_argument('--wandb', action="store_true",
                        help="Enable wandb logging")
    return parser.parse_args()


prompt_template = """Answer this question based on retrieved contexts. Just give the answer without explanation.
    [QUESTION]
    {question}
    [END_QUESTION]

    [RETRIEVED_CONTEXTS]
    {context}
    [END_RETRIEVED_CONTEXTS]

    Answer: """


def _process_and_chunk_text(text, chunk_size):
    full_chunks = []
    # Some corpora of docs are grouped into chunked files, typically by paragraph.
    # Only split into individual documents if many paragraphs are detected
    paragraphs = re.split(r'\n{2,}', text)
    if len(paragraphs) < 16:
        paragraphs = [text]

    for paragraph in paragraphs:
        chunks = _chunk_text(paragraph, chunk_size)
        full_chunks.extend(chunks)
    return full_chunks


def get_data(args):
    # need a JSON dict of Questions and answers, see below for how its used
    with open('train.json') as file:
        json_obj = json.load(file)
    text_contexts = []

    # Read corpus data to create the KG. Prefer *.json files, fall back to txt
    # files.
    # TODO: add support for additional corpus file formats: PDF, CSV, XML,
    # HTML, possibly others.
    file_paths = glob(f"corpus/*.json")
    if len(file_paths) > 0:
        for file_path in file_paths:
            with open(file_path, "r+") as f:
                data = json.load(f)
            doc_type = data[0]["document_type"]
            if doc_type != "text":
                raise ValueError(f"Bad extraction for {file_path}, expecting "
                                 f"text only but got {doc_type}")
            text_contexts.extend(
                _process_and_chunk_text(data[0]["metadata"]["content"],
                                        args.doc_chunk_size))
    else:
        for file_path in glob(f"corpus/*"):
            with open(file_path, "r+") as f:
                text_context = f.read()
            text_contexts.extend(
                _process_and_chunk_text(text_context, args.doc_chunk_size))

    return json_obj, text_contexts


def make_dataset(args):
    if os.path.exists("tech_qa.pt"):
        print("Re-using Saved TechQA KG-RAG Dataset...")
        return torch.load("tech_qa.pt", weights_only=False)
    else:
        qa_pairs, context_docs = get_data(args)
        print("Number of Docs in our VectorDB =", len(context_docs))
        data_lists = {"train": [], "validation": [], "test": []}
        triples = []
        if os.path.exists("tech_qa_just_triples.pt"):
            triples = torch.load("tech_qa_just_triples.pt", weights_only=False)
        else:
            kg_maker = TXT2KG(NVIDIA_NIM_MODEL=args.NV_NIM_MODEL,
                              NVIDIA_API_KEY=args.NV_NIM_KEY,
                              ENDPOINT_URL=args.ENDPOINT_URL,
                              chunk_size=args.chunk_size)
            print(
                "Note that if the TXT2KG process is too slow for you're liking using the public NIM, consider deploying yourself using local_lm flag of TXT2KG or using https://build.nvidia.com/nvidia/llama-3_1-nemotron-70b-instruct?snippet_tab=Docker to deploy to a private endpoint, which you can pass to this script w/ --ENDPOINT_URL flag."
            )  # noqa
            total_tqdm_count = len(context_docs)
            initial_tqdm_count = 0
            if os.path.exists("checkpoint_kg.pt"):
                print("Restoring KG from checkpoint...")
                saved_relevant_triples = torch.load("checkpoint_kg.pt",
                                                    weights_only=False)
                kg_maker.relevant_triples = saved_relevant_triples
                kg_maker.doc_id_counter = len(saved_relevant_triples)
                initial_tqdm_count = kg_maker.doc_id_counter
                context_docs = context_docs[kg_maker.doc_id_counter:]

            chkpt_interval = 10
            chkpt_count = 0
            for context_doc in tqdm(context_docs, total=total_tqdm_count,
                                    initial=initial_tqdm_count,
                                    desc="Extracting KG triples"):
                kg_maker.add_doc_2_KG(txt=context_doc)
                chkpt_count += 1
                if chkpt_count == chkpt_interval:
                    chkpt_count = 0
                    kg_maker.save_kg("checkpoint_kg.pt")
            relevant_triples = kg_maker.relevant_triples

            triples.extend(
                list(
                    chain.from_iterable(
                        triple_set
                        for triple_set in relevant_triples.values())))
            triples = list(dict.fromkeys(triples))
            torch.save(triples, "tech_qa_just_triples.pt")
            if os.path.exists("checkpoint_kg.pt"):
                os.remove("checkpoint_kg.pt")

        print("Number of triples in our GraphDB =", len(triples))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sent_trans_batch_size = 256
        model = SentenceTransformer(
            model_name='Alibaba-NLP/gte-modernbert-base').to(device)
        fs, gs = create_remote_backend_from_triplets(
            triplets=triples, node_embedding_model=model,
            node_method_to_call="encode", path="backend",
            pre_transform=preprocess_triplet, node_method_kwargs={
                "batch_size": min(len(triples), sent_trans_batch_size),
                "verbose": True,
            }, graph_db=NeighborSamplingRAGGraphStore,
            feature_db=ModernBertFeatureStore).load()
        """
        NOTE: these retriever hyperparams are very important.
        Tuning may be needed for custom data...
        """
        # encode the raw context docs
        embedded_docs = model.encode(context_docs, output_device=device,
                                     batch_size=int(sent_trans_batch_size / 4),
                                     verbose=True)
        # k for KNN
        knn_neighsample_bs = 1024
        # number of neighbors for each seed node selected by KNN
        fanout = 100
        # number of hops for neighborsampling
        num_hops = 2
        local_filter_kwargs = {
            "topk": 5,  # nodes
            "topk_e": 5,  # edges
            "cost_e": .5,  # edge cost
            "num_clusters": 10,  # num clusters
        }
        print("Now to retrieve context for each query from\
            our Vector and Graph DBs...")
        # GraphDB retrieval done with KNN+NeighborSampling+PCST
        # PCST = Prize Collecting Steiner Tree
        # VectorDB retrieval just vanilla RAG
        # TODO add reranking NIM to VectorRAG
        query_loader = RAGQueryLoader(
            data=(fs, gs), seed_nodes_kwargs={"k_nodes": knn_neighsample_bs},
            sampler_kwargs={"num_neighbors": [fanout] * num_hops},
            local_filter=make_pcst_filter(triples, model),
            local_filter_kwargs=local_filter_kwargs, raw_docs=context_docs,
            embedded_docs=embedded_docs, k_for_docs=args.k_for_docs)
        total_data_list = []
        extracted_triple_sizes = []
        for data_point in tqdm(qa_pairs, desc="Building un-split dataset"):
            if data_point["is_impossible"]:
                continue
            QA_pair = (data_point["question"], data_point["answer"])
            q = QA_pair[0]
            subgraph = query_loader.query(q)
            subgraph.label = QA_pair[1]
            total_data_list.append(subgraph)
            extracted_triple_sizes.append(len(subgraph.triples))
        random.shuffle(total_data_list)
        print("Min # of Retrieved Triples =", min(extracted_triple_sizes))
        print("Max # of Retrieved Triples =", max(extracted_triple_sizes))
        print("Average # of Retrieved Triples =",
              sum(extracted_triple_sizes) / len(extracted_triple_sizes))

        # 60:20:20 split
        data_lists["train"] = total_data_list[:int(.6 * len(total_data_list))]
        data_lists["validation"] = total_data_list[
            int(.6 * len(total_data_list)):int(.8 * len(total_data_list))]
        data_lists["test"] = total_data_list[int(.8 * len(total_data_list)):]

        torch.save(data_lists, "tech_qa.pt")
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return data_lists


def train(args, data_lists):
    if args.wandb:
        wandb.init(project=args.wandb_project,
                   name=f"run_{datetime.now().strftime('%Y-%m-%d_%H:%M')}",
                   config=vars(args))

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
    gnn = GAT(in_channels=768, hidden_channels=hidden_channels,
              out_channels=1024, num_layers=num_gnn_layers, heads=4)
    if args.llm_generator_mode == "full":
        llm = LLM(model_name=args.llm_generator_name)
        model = GRetriever(llm=llm, gnn=gnn)
    elif args.llm_generator_mode == "lora":
        llm = LLM(model_name=args.llm_generator_name, dtype=torch.float32)
        model = GRetriever(llm=llm, gnn=gnn, use_lora=True)
    else:
        # frozen
        llm = LLM(model_name=args.llm_generator_name,
                  dtype=torch.float32).eval()
        for _, p in llm.named_parameters():
            p.requires_grad = False
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
        for epoch in range(args.epochs):
            model.train()
            epoch_loss = 0
            epoch_str = f'Epoch: {epoch + 1}|{args.epochs}'
            loader = tqdm(train_loader, desc=epoch_str)
            for step, batch in enumerate(loader):
                new_qs = []
                for i, q in enumerate(batch["question"]):
                    # insert VectorRAG context
                    new_qs.append(
                        prompt_template.format(question=q,
                                               context=batch.text_context[i]))
                batch.question = new_qs

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

                if args.wandb and (step + 1) % args.log_steps == 0:
                    wandb.log({
                        "train/loss": float(loss),
                        "train/lr": optimizer.param_groups[0]['lr'],
                    })

                if (step + 1) % 2 == 0:
                    lr = optimizer.param_groups[0]['lr']
            train_loss = epoch_loss / len(train_loader)
            print(epoch_str + f', Train Loss: {train_loss:4f}')

            # Eval Step
            val_loss = 0
            model.eval()
            with torch.no_grad():
                for step, batch in enumerate(val_loader):
                    loss = get_loss(model, batch)
                    val_loss += loss.item()
                val_loss = val_loss / len(val_loader)
                print(epoch_str + f", Val Loss: {val_loss:4f}")

                if args.wandb:
                    wandb.log({
                        "val/loss": val_loss,
                        "train/epoch_loss": train_loss,
                        "epoch": epoch + 1
                    })

        if args.wandb:
            wandb.finish()

        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        model.eval()
        if not args.dont_save_model:
            save_params_dict(model, save_path=save_name)
    return model, test_loader


def test(model, test_loader, args):
    llm_judge = LLMJudge(args.NV_NIM_MODEL, args.NV_NIM_KEY, args.ENDPOINT_URL)

    def eval(question: str, pred: str, correct_answer: str):
        # calculate the score based on pred and correct answer
        return llm_judge.score(question, pred, correct_answer)

    scores = []
    eval_tuples = []
    for test_batch in tqdm(test_loader, desc="Testing"):
        new_qs = []
        for i, q in enumerate(test_batch["question"]):
            # insert VectorRAG context
            new_qs.append(
                prompt_template.format(question=q,
                                       context=test_batch.text_context[i]))
        test_batch.question = new_qs
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
    if args.wandb and not wandb_available:
        print("Error: wandb package not found but --wandb flag was used.")
        print("Please install wandb and rerun the script.")
        sys.exit(1)
    data_lists = make_dataset(args)
    model, test_loader = train(args, data_lists)
    test(model, test_loader, args)
