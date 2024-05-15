"""This example implements G-retriever using PyG.
Original Paper: https://arxiv.org/abs/2402.07630
“G-Retriever significantly reduces hallucinations
by 54% compared to the [LLM] baseline“.
requirements on top of basic PyG:
pip install datasets transformers pcst_fast sentencepiece tqdm pandas
"""
import argparse
import gc
import time
from os import path

import torch
from g_retriever import minimal_demo, train

from torch_geometric import seed_everything
from torch_geometric.datasets import SQUAD_WikiGraph


def get_list_of_embeddings_from_batch(batch):
    rag_embeddings = []
    ptr = batch.ptr
    for i in range(len(ptr) - 1):
        if ptr[i] - ptr[i + 1] > 0:
            rag_embeddings.append(batch.x[ptr[i]:ptr[i + 1]])
        else:
            rag_embeddings.append(None)
    return rag_embeddings


def get_loss(model, batch, model_save_name) -> torch.Tensor:
    if model_save_name == "llm":
        rag_embeddings = get_list_of_embeddings_from_batch(batch)
        return model(batch.question, batch.label, rag_embeddings)
    else:
        return model(batch.question, batch.x, batch.edge_index, batch.batch,
                     batch.ptr, batch.label, batch.edge_attr)


def inference_step(model, batch, model_save_name):
    if model_save_name == "llm":
        rag_embeddings = get_list_of_embeddings_from_batch(batch)
        return model.inference(batch.question, rag_embeddings)
    else:
        return model.inference(batch.question, batch.x, batch.edge_index,
                               batch.batch, batch.ptr, batch.edge_attr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gnn_hidden_channels', type=int, default=1024)
    parser.add_argument('--num_gnn_layers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--eval_batch_size', type=int, default=8)
    parser.add_argument(
        "--checkpointing", action="store_true",
        help="Use this flag to checkpoint each time a \
        new best val loss is achieved")

    args = parser.parse_args()
    # check if saved model
    retrain = True
    if path.exists("gnn_llm.pt") and path.exists("gnn_llm_eval_outs.pt"):
        print("Existing trained model found.")
        print("Would you like to retrain?")
        user_input = str(input("(y/n):")).lower()
        retrain = user_input == "y"
    else:
        retrain = True
    if retrain:
        since = time.time()
        prep_time, dataset, gnn_llm_eval_outs = train(
            since, args.epochs, args.gnn_hidden_channels, args.num_gnn_layers,
            args.batch_size, args.eval_batch_size, args.lr, get_loss,
            inference_step, dataset=SQUAD_WikiGraph(), checkpointing=args.checkpointing)
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        gc.collect()
        e2e_time = round(time.time() - since, 2)
        print("E2E time (e2e_time) =", e2e_time, "seconds")
        print("E2E tme minus Prep Time =", e2e_time - prep_time, "seconds")
    else:
        gnn_llm_eval_outs = torch.load("gnn_llm_eval_outs.pt")
        dataset = SQUAD_WikiGraph()
    print("Here's a demo showcasing how GNN reduces LLM hallucinations:")
    minimal_demo(gnn_llm_eval_outs, dataset, args.lr, args.epochs,
                 args.batch_size, args.eval_batch_size, get_loss,
                 inference_step, skip_pretrained_llm=True)
