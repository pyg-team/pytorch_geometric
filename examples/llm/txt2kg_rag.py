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
from pathlib import Path

import yaml

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
from huggingface_hub import hf_hub_download
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from torch_geometric import seed_everything
from torch_geometric.llm import RAGQueryLoader
from torch_geometric.llm.models import (
    LLM,
    TXT2KG,
    GRetriever,
    LLMJudge,
    SentenceTransformer,
)
from torch_geometric.llm.models.txt2kg import _chunk_text
from torch_geometric.llm.utils.backend_utils import (
    create_graph_from_triples,
    create_remote_backend_from_graph_data,
    make_pcst_filter,
    preprocess_triplet,
)
from torch_geometric.llm.utils.feature_store import KNNRAGFeatureStore
from torch_geometric.llm.utils.graph_store import NeighborSamplingRAGGraphStore
from torch_geometric.llm.utils.vectorrag import DocumentRetriever
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GAT, SGFormer

# Define constants for better readability
NV_NIM_MODEL_DEFAULT = "nvidia/llama-3.1-nemotron-ultra-253b-v1"
LLM_GENERATOR_NAME_DEFAULT = "meta-llama/Meta-Llama-3.1-8B-Instruct"
ENCODER_MODEL_NAME_DEFAULT = "Alibaba-NLP/gte-modernbert-base"
KG_CHUNK_SIZE_DEFAULT = 512
GNN_HID_CHANNELS_DEFAULT = 1024
GNN_LAYERS_DEFAULT = 4
LR_DEFAULT = 1e-5
EPOCHS_DEFAULT = 2
BATCH_SIZE_DEFAULT = 1
EVAL_BATCH_SIZE_DEFAULT = 2
LLM_GEN_MODE_DEFAULT = "full"
DEFAULT_ENDPOINT_URL = "https://integrate.api.nvidia.com/v1"
max_chars_in_train_answer = 128


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gnn_model', type=str, default="GAT",
                        choices=["GAT", "SGFormer"],
                        help="The GNN model to use. Default is GAT.")
    parser.add_argument('--NV_NIM_MODEL', type=str,
                        default=NV_NIM_MODEL_DEFAULT,
                        help="The NIM LLM to use for TXT2KG for LLMJudge")
    parser.add_argument('--NV_NIM_KEY', type=str, help="NVIDIA API key")
    parser.add_argument(
        '--ENDPOINT_URL', type=str, default=DEFAULT_ENDPOINT_URL,
        help="The URL hosting your model, \
        in case you are not using the public NIM.")
    parser.add_argument(
        '--kg_chunk_size', type=int, default=KG_CHUNK_SIZE_DEFAULT,
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
    parser.add_argument(
        '--llm_generator_mode', type=str, default=LLM_GEN_MODE_DEFAULT,
        choices=["frozen", "lora",
                 "full"], help="Whether to freeze the Generator LLM,\
                        use LORA, or fully finetune")
    parser.add_argument('--dont_save_model', action="store_true",
                        help="Whether to skip model saving.")
    parser.add_argument('--log_steps', type=int, default=30,
                        help="Log to wandb every N steps")
    parser.add_argument('--wandb_project', type=str, default="techqa",
                        help="Weights & Biases project name")
    parser.add_argument('--wandb', action="store_true",
                        help="Enable wandb logging")
    parser.add_argument(
        '--num_gpus', type=int, default=None,
        help="Number of GPUs to use. If not specified,"
        "will determine automatically based on model size.")
    parser.add_argument('--regenerate_dataset', action="store_true",
                        help="Regenerate the dataset")
    parser.add_argument(
        '--doc_parsing_mode', type=str, default=None,
        choices=["paragraph",
                 "file"], help="How to parse documents: 'paragraph' splits "
        "files by paragraphs, 'file' treats each file as"
        "one document. "
        "This will override any value set in the config file.")
    parser.add_argument(
        '--k_for_docs', type=int, default=None,
        help="Number of docs to retrieve for each question. "
        "This will override any value set in the config file.")
    parser.add_argument(
        '--doc_chunk_size', type=int, default=None,
        help="The chunk size to use VectorRAG (document retrieval). "
        "This will override any value set in the config file.")
    parser.add_argument(
        '--dataset', type=str, default="techqa", help="Dataset folder name, "
        "should contain corpus and train.json files."
        "extracted triples, processed dataset, "
        "document retriever, and model checkpoints "
        "will be saved in the dataset folder")
    parser.add_argument(
        '--skip_graph_rag', action="store_true",
        help="Skip the graph RAG step. "
        "Used to compare the performance of Vector+Graph RAG vs Vector RAG.")
    parser.add_argument(
        '--use_x_percent_corpus', default=100.0, type=float,
        help="Debug flag that allows user to only use a random percentage "
        "of available knowledge base corpus for RAG")
    args = parser.parse_args()

    assert args.NV_NIM_KEY, "NVIDIA API key is required for TXT2KG and eval"
    assert args.use_x_percent_corpus <= 100 and \
        args.use_x_percent_corpus > 0, "Please provide a value in (0,100]"
    if args.skip_graph_rag:
        print("Skipping graph RAG step, setting GNN layers to 0...")
        args.num_gnn_layers = 0

    config_path = os.path.join(args.dataset, "config.yaml")
    if os.path.exists(config_path):
        print(f"Loading config from {config_path}...")
        with open(config_path) as config_file:
            config = yaml.safe_load(config_file)

        if config is not None:
            # Use a loop to check and apply config values for each parameter
            config_params = [
                'doc_parsing_mode', 'doc_chunk_size', 'k_for_docs'
            ]
            for param in config_params:
                if param in config and getattr(args, param) is None:
                    setattr(args, param, config[param])
                    print(f"Using config value for {param}: {config[param]}")
    else:
        print("Skipping config loading...")
        if args.dataset == "techqa":
            if args.doc_chunk_size is None:
                args.doc_chunk_size = 1024
            if args.k_for_docs is None:
                args.k_for_docs = 14

    assert args.doc_chunk_size is not None, "doc_chunk_size has not been set"
    assert args.k_for_docs is not None, "k_for_docs has not been set"

    return args


sys_prompt = (
    "You are an expert assistant that can answer "
    "any question from its knowledge, given a knowledge graph embedding and "
    "it's textualized context. Just give the answer, without explanation.")

prompt_template = """
    [QUESTION]
    {question}
    [END_QUESTION]

    [RETRIEVED_CONTEXTS]
    {context}
    [END_RETRIEVED_CONTEXTS]
    """


def _process_and_chunk_text(text, chunk_size, doc_parsing_mode):
    full_chunks = []
    """
    Some corpora of docs are grouped into chunked files,
    typically by paragraph.
    Only split into individual documents
    if multiple paragraphs are detected.
    """
    if doc_parsing_mode == "paragraph":
        paragraphs = re.split(r'\n{2,}', text)
    else:
        # doc_parsing_mode == 'file' or doc_parsing_mode is None
        paragraphs = [text]

    for paragraph in paragraphs:
        if chunk_size is not None:
            chunks = _chunk_text(paragraph, chunk_size)
        else:
            # defaults to 512 in _chunk_text
            chunks = _chunk_text(paragraph)
        full_chunks.extend(chunks)
    return full_chunks


def get_data(args):
    # need a JSON dict of Questions and answers, see below for how its used

    json_path = Path(args.dataset) / "train.json"
    corpus_path = Path(args.dataset) / "corpus"

    # techqa specified but neither corpus or train.json exists
    if "techqa" in args.dataset.lower() and not (json_path.exists()
                                                 or corpus_path.exists()):
        print("Could not find Q&A pairs and/or knowledge base corpus")
        print("Would you like to download the TechQA dataset for demo?")
        user_input = input("Y/N: ")
        if user_input.lower() == "y" or user_input.lower() == "yes":
            print("Downloading data...")
            # downloads
            zip_path = hf_hub_download(
                repo_id="nvidia/TechQA-RAG-Eval",
                repo_type="dataset",
                filename="corpus.zip",
            )
            json_path = hf_hub_download(
                repo_id="nvidia/TechQA-RAG-Eval",
                repo_type="dataset",
                filename="train.json",
            )
            # move to working dir
            if not os.path.exists(args.dataset):
                os.mkdir(args.dataset)
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(args.dataset)
            import shutil
            shutil.copy(json_path, os.path.join(args.dataset, "train.json"))
        elif user_input.lower() == "n" or user_input.lower() == "no":
            sys.exit("No selected, no data to work with... exiting.")
        else:
            sys.exit("Invalid user input, exiting.")
    with open(os.path.join(args.dataset, "train.json")) as file:
        json_obj = json.load(file)
    text_contexts = []

    # Read corpus data to create the KG and for document retrieval (RAG).
    # Prefer *.json files, fall back to txt files.
    # TODO: add support for additional corpus file formats: PDF, CSV, XML,
    # HTML, possibly others.
    # corpus folder is simply a folder with context documents in it.
    file_paths = glob(os.path.join(args.dataset, "corpus", "*.json"))
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
                                        args.doc_chunk_size,
                                        args.doc_parsing_mode))
    else:
        for file_path in glob(os.path.join(args.dataset, "corpus", "*")):
            with open(file_path, "r+") as f:
                text_context = f.read()
            text_contexts.extend(
                _process_and_chunk_text(text_context, args.doc_chunk_size,
                                        args.doc_parsing_mode))
    if args.use_x_percent_corpus < 100:
        random.shuffle(text_contexts)
        text_contexts = text_contexts[
            0:int(len(text_contexts) * args.use_x_percent_corpus / 100.0)]

    return json_obj, text_contexts


def index_kg(args, context_docs):
    kg_maker = TXT2KG(NVIDIA_NIM_MODEL=args.NV_NIM_MODEL,
                      NVIDIA_API_KEY=args.NV_NIM_KEY,
                      ENDPOINT_URL=args.ENDPOINT_URL,
                      chunk_size=args.kg_chunk_size)
    print(
        "Note that if the TXT2KG process is too slow for you're liking using "
        "the public NIM, consider deploying yourself using local_lm flag of "
        "TXT2KG or using https://build.nvidia.com/nvidia/llama-3_1-nemotron-70b-instruct "  # noqa
        "to deploy to a private endpoint, which you can pass to this script "
        "w/ --ENDPOINT_URL flag.")
    print(
        "Guide for deploying NIM: https://developer.nvidia.com/blog/a-simple-guide-to-deploying-generative-ai-with-nvidia-nim/"  # noqa
    )
    total_tqdm_count = len(context_docs)
    initial_tqdm_count = 0
    checkpoint_file = list(Path(args.dataset).glob("*--*--checkpoint_kg.pt"))
    if len(checkpoint_file) > 1:
        raise RuntimeError("Error: more than one checkpoint file found")

    if len(checkpoint_file) == 1:
        print("Restoring KG from checkpoint")
        checkpoint_file = checkpoint_file[0]
        checkpoint_model_name = checkpoint_file.name.split('--')[0]
        # check if triples generation are using the correct model
        if args.NV_NIM_MODEL.split('/')[-1] != checkpoint_model_name:
            raise RuntimeError(
                "Error: stored triples were generated using a different model")
        saved_relevant_triples = torch.load(checkpoint_file,
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
            path = args.dataset + "/{m}--{t}--checkpoint_kg.pt"
            model = kg_maker.NIM_MODEL.split(
                '/')[-1] if not kg_maker.local_LM else "local"
            path = path.format(m=model,
                               t=datetime.now().strftime("%Y%m%d_%H%M%S"))
            torch.save(kg_maker.relevant_triples, path)

    relevant_triples = kg_maker.relevant_triples
    triples = list(
        chain.from_iterable(triple_set
                            for triple_set in relevant_triples.values()))
    triples = list(dict.fromkeys(triples))
    raw_triples_path = args.dataset + "/{m}--{t}--raw_triples.pt"

    model_name = kg_maker.NIM_MODEL.split(
        '/')[-1] if not kg_maker.local_LM else "local"
    torch.save(
        triples,
        raw_triples_path.format(m=model_name,
                                t=datetime.now().strftime("%Y%m%d_%H%M%S")))

    for old_checkpoint_file in Path(
            args.dataset).glob("*--*--checkpoint_kg.pt"):
        os.remove(old_checkpoint_file)

    return triples


def update_data_lists(args, data_lists):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # creating the embedding model
    sent_trans_batch_size = 256
    model = SentenceTransformer(
        model_name=ENCODER_MODEL_NAME_DEFAULT).to(device).eval()
    model_kwargs = {
        "output_device": device,
        "batch_size": int(sent_trans_batch_size / 4),
    }
    doc_retriever_path = os.path.join(args.dataset, "document_retriever.pt")
    if os.path.exists(doc_retriever_path):
        print("Loading document retriever from checkpoint...")
        vector_retriever = DocumentRetriever.load(doc_retriever_path,
                                                  model=model.encode,
                                                  model_kwargs=model_kwargs)
        if args.k_for_docs != vector_retriever.k_for_docs:
            vector_retriever.k_for_docs = args.k_for_docs
        else:
            return data_lists
    else:
        raise ValueError("Document retriever not found")

    print("k_for_docs changed, updating data lists...")

    total_points = sum(len(data_list) for data_list in data_lists.values())

    progress_bar = tqdm(total=total_points, desc="Updating text contexts")

    for data_list in data_lists.values():
        for data_point in data_list:
            q = data_point["question"]
            data_point["text_context"] = vector_retriever.query(q)
            progress_bar.update(1)

    progress_bar.close()

    vector_retriever.save(doc_retriever_path)

    del vector_retriever
    gc.collect()
    torch.cuda.empty_cache()

    dataset_name = os.path.basename(args.dataset)
    dataset_path = os.path.join(args.dataset, f"{dataset_name}.pt")
    torch.save(data_lists, dataset_path)
    return data_lists


def make_dataset(args):
    qa_pairs, context_docs = get_data(args)
    print("Number of Docs in our VectorDB =", len(context_docs))
    data_lists = {"train": [], "validation": [], "test": []}

    triples = []
    raw_triples_file = list(Path(args.dataset).glob("*--*--raw_triples.pt"))
    if len(raw_triples_file) > 1:
        raise RuntimeError("Error: multiple raw_triples files found")
    if len(raw_triples_file) == 1:
        raw_triples_file = raw_triples_file[0]
        stored_model_name = raw_triples_file.name.split('--')[0]

        if args.NV_NIM_MODEL.split('/')[-1] != stored_model_name:
            raise RuntimeError(
                "Error: stored triples were generated using a different model")

        print(f" -> Saved triples generated with: {stored_model_name}")
        triples = torch.load(raw_triples_file)
    else:
        triples = index_kg(args, context_docs)

    print("Number of triples in our GraphDB =", len(triples))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # creating the embedding model
    sent_trans_batch_size = 256
    model = SentenceTransformer(
        model_name=ENCODER_MODEL_NAME_DEFAULT).to(device)

    print("Creating the graph data from raw triples...")
    # create the graph data from raw triples
    graph_data = create_graph_from_triples(
        triples=triples, embedding_model=model.encode,
        embedding_method_kwargs={
            "batch_size": min(len(triples), sent_trans_batch_size),
            "verbose": True
        }, pre_transform=preprocess_triplet)

    print("Creating the graph and feature stores...")
    # creating the graph and feature stores
    fs, gs = create_remote_backend_from_graph_data(
        graph_data=graph_data, path="backend",
        graph_db=NeighborSamplingRAGGraphStore,
        feature_db=KNNRAGFeatureStore).load()
    """
    NOTE: these retriever hyperparams are very important.
    Tuning may be needed for custom data...
    """

    model_kwargs = {
        "output_device": device,
        "batch_size": int(sent_trans_batch_size / 4),
        "verbose": True
    }

    doc_retriever_path = os.path.join(args.dataset, "document_retriever.pt")
    if os.path.exists(doc_retriever_path):
        print("Loading document retriever from checkpoint...")
        vector_retriever = DocumentRetriever.load(doc_retriever_path,
                                                  model=model.encode,
                                                  model_kwargs=model_kwargs)
        if args.k_for_docs != vector_retriever.k_for_docs:
            vector_retriever.k_for_docs = args.k_for_docs
    else:
        print("Creating document retriever...")
        vector_retriever = DocumentRetriever(context_docs,
                                             k_for_docs=args.k_for_docs,
                                             model=model.encode,
                                             model_kwargs=model_kwargs)
        vector_retriever.save(doc_retriever_path)

    subgraph_filter = make_pcst_filter(
        triples,
        model,
        topk=5,  # nodes
        topk_e=5,  # edges
        cost_e=.5,  # edge cost
        num_clusters=10)  # num clusters

    # number of neighbors for each seed node selected by KNN
    fanout = 100
    # number of hops for neighborsampling
    num_hops = 2

    query_loader_config = {
        "k_nodes": 1024,  # k for Graph KNN
        "num_neighbors": [fanout] * num_hops,  # number of sampled neighbors
        "encoder_model": model,
    }

    # GraphDB retrieval done with KNN+NeighborSampling+PCST
    # PCST = Prize Collecting Steiner Tree
    # VectorDB retrieval just vanilla vector RAG
    print("Now to retrieve context for each query from "
          "our Vector and Graph DBs...")

    query_loader = RAGQueryLoader(graph_data=(fs, gs),
                                  subgraph_filter=subgraph_filter,
                                  vector_retriever=vector_retriever,
                                  config=query_loader_config)

    # pre-process the dataset
    total_data_list = []
    extracted_triple_sizes = []
    global max_chars_in_train_answer
    for data_point in tqdm(qa_pairs, desc="Building un-split dataset"):
        if data_point["is_impossible"]:
            continue
        QA_pair = (data_point["question"], data_point["answer"])
        q = QA_pair[0]
        max_chars_in_train_answer = max(len(QA_pair[1]),
                                        max_chars_in_train_answer)
        # (TODO) make this batch queries for retrieving w/ CuVS+CuGraph
        subgraph = query_loader.query(q)
        subgraph.label = QA_pair[1]
        total_data_list.append(subgraph)
        extracted_triple_sizes.append(len(subgraph.triples))
    random.shuffle(total_data_list)

    # stats
    print("Min # of Retrieved Triples =", min(extracted_triple_sizes))
    print("Max # of Retrieved Triples =", max(extracted_triple_sizes))
    print("Average # of Retrieved Triples =",
          sum(extracted_triple_sizes) / len(extracted_triple_sizes))

    # 60:20:20 split
    data_lists["train"] = total_data_list[:int(.6 * len(total_data_list))]
    data_lists["validation"] = total_data_list[int(.6 * len(total_data_list)
                                                   ):int(.8 *
                                                         len(total_data_list))]
    data_lists["test"] = total_data_list[int(.8 * len(total_data_list)):]

    dataset_name = os.path.basename(args.dataset)
    dataset_path = os.path.join(args.dataset, f"{dataset_name}.pt")
    torch.save((data_lists, max_chars_in_train_answer), dataset_path)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return data_lists


def train(args, train_loader, val_loader):
    if args.wandb:
        wandb.init(project=args.wandb_project,
                   name=f"run_{datetime.now().strftime('%Y-%m-%d_%H:%M')}",
                   config=vars(args))
    hidden_channels = args.gnn_hidden_channels
    num_gnn_layers = args.num_gnn_layers

    if args.num_gnn_layers > 0:
        if args.gnn_model == "GAT":
            gnn = GAT(in_channels=768, hidden_channels=hidden_channels,
                      out_channels=1024, num_layers=num_gnn_layers, heads=4)
        elif args.gnn_model == "SGFormer":
            gnn = SGFormer(in_channels=768, hidden_channels=hidden_channels,
                           out_channels=1024, trans_num_heads=1,
                           trans_dropout=0.5, gnn_num_layers=num_gnn_layers,
                           gnn_dropout=0.5)
        else:
            raise ValueError(f"Invalid GNN model: {args.gnn_model}")
    else:
        gnn = None

    if args.llm_generator_mode == "full":
        llm = LLM(model_name=args.llm_generator_name, sys_prompt=sys_prompt,
                  n_gpus=args.num_gpus)
    elif args.llm_generator_mode == "lora":
        llm = LLM(model_name=args.llm_generator_name, sys_prompt=sys_prompt,
                  dtype=torch.float32, n_gpus=args.num_gpus)
    else:
        # frozen
        llm = LLM(model_name=args.llm_generator_name, sys_prompt=sys_prompt,
                  dtype=torch.float32, n_gpus=args.num_gpus).eval()
        for _, p in llm.named_parameters():
            p.requires_grad = False

    model = GRetriever(llm=llm, gnn=gnn,
                       use_lora=args.llm_generator_mode == "lora")

    save_name = os.path.join(args.dataset, "model.pt")

    if args.llm_generator_mode == "frozen" and args.num_gnn_layers == 0:
        if not args.dont_save_model:
            save_params_dict(model, save_path=save_name)
        return model

    if os.path.exists(save_name) and not args.regenerate_dataset:
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

        num_oom_errors = 0
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
                        prompt_template.format(
                            question=q,
                            context="\n".join(batch.text_context[i])))
                batch.question = new_qs

                if args.skip_graph_rag:
                    batch.desc = ""

                optimizer.zero_grad()
                try:
                    loss = get_loss(model, batch)
                    loss.backward()
                    clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1)
                    if (step + 1) % 2 == 0:
                        adjust_learning_rate(optimizer.param_groups[0], lr,
                                             step / len(train_loader) + epoch,
                                             args.epochs)
                    optimizer.step()
                    epoch_loss += float(loss.detach())

                    if args.wandb and (step + 1) % args.log_steps == 0:
                        wandb.log({
                            "train/loss": float(loss.detach()),
                            "train/lr": optimizer.param_groups[0]['lr'],
                        })

                    if (step + 1) % 2 == 0:
                        lr = optimizer.param_groups[0]['lr']
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print("Sequence length of last batch: ",
                          model.seq_length_stats[-1])
                    # TODO: Implement CPU fallback (WIP)
                    num_oom_errors += 1
            print("Sequence length stats: ")
            print("seq_len avg: ",
                  sum(model.seq_length_stats) / len(model.seq_length_stats))
            print("seq_len min: ", min(model.seq_length_stats))
            print("seq_len max: ", max(model.seq_length_stats))
            print("Percent of OOM errors: ",
                  num_oom_errors / len(train_loader))
            train_loss = epoch_loss / len(train_loader)
            print(epoch_str + f', Train Loss: {train_loss:4f}')

            # Eval Step
            val_loss = 0
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    new_qs = []
                    for i, q in enumerate(batch["question"]):
                        # insert VectorRAG context
                        new_qs.append(
                            prompt_template.format(
                                question=q,
                                context="\n".join(batch.text_context[i])))
                    batch.question = new_qs
                    if args.skip_graph_rag:
                        batch.desc = ""
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
    return model


def test(model, test_loader, args):
    llm_judge = LLMJudge(args.NV_NIM_MODEL, args.NV_NIM_KEY, args.ENDPOINT_URL)

    def eval(question: str, pred: str, correct_answer: str):
        # calculate the score based on pred and correct answer
        return llm_judge.score(question, pred, correct_answer)

    scores = []
    eval_tuples = []
    for test_batch in tqdm(test_loader, desc="Testing"):
        new_qs = []
        raw_qs = test_batch["question"]
        for i, q in enumerate(test_batch["question"]):
            # insert VectorRAG context
            new_qs.append(
                prompt_template.format(
                    question=q, context="\n".join(test_batch.text_context[i])))
        test_batch.question = new_qs
        if args.skip_graph_rag:
            test_batch.desc = ""
        preds = (inference_step(model, test_batch,
                                max_out_tokens=max_chars_in_train_answer / 2))
        for question, pred, label in zip(raw_qs, preds, test_batch.label):
            eval_tuples.append((question, pred, label))
    for question, pred, label in tqdm(eval_tuples, desc="Eval"):
        scores.append(eval(question, pred, label))
    avg_scores = sum(scores) / len(scores)
    print("Avg marlin accuracy=", avg_scores)
    print("*" * 5 + "NOTE" + "*" * 5)
    print("Marlin Accuracy is Estimated by LLM as a Judge!")
    print("Improvement of this estimation process is WIP...")


if __name__ == '__main__':
    # for reproducibility
    seed_everything(50)

    args = parse_args()
    if args.wandb and not wandb_available:
        print("Error: wandb package not found but --wandb flag was used.")
        print("Please install wandb and rerun the script.")
        sys.exit(1)

    # Need to sanitize sensitive keys
    saved_NIM_KEY = args.NV_NIM_KEY
    args.NV_NIM_KEY = "********"
    print(f"Starting {args.dataset} training with args: ", args)
    args.NV_NIM_KEY = saved_NIM_KEY

    dataset_name = os.path.basename(args.dataset)
    dataset_path = os.path.join(args.dataset, f"{dataset_name}.pt")
    if os.path.exists(dataset_path) and not args.regenerate_dataset:
        print(f"Re-using Saved {dataset_name} KG-RAG Dataset...")
        data_lists, max_chars_in_train_answer = torch.load(
            dataset_path, weights_only=False)
        doc_retriever_path = os.path.join(args.dataset,
                                          "document_retriever.pt")
        if os.path.exists(doc_retriever_path):
            print("Updating data lists with document retriever...")
            data_lists = update_data_lists(args, data_lists)
    else:
        data_lists = make_dataset(args)
    batch_size = args.batch_size
    eval_batch_size = args.eval_batch_size
    train_loader = DataLoader(data_lists["train"], batch_size=batch_size,
                              drop_last=True, pin_memory=True, shuffle=True)
    val_loader = DataLoader(data_lists["validation"],
                            batch_size=eval_batch_size, drop_last=False,
                            pin_memory=True, shuffle=False)
    test_loader = DataLoader(data_lists["test"], batch_size=eval_batch_size,
                             drop_last=False, pin_memory=True, shuffle=False)

    model = train(args, train_loader, val_loader)
    test(model, test_loader, args)
