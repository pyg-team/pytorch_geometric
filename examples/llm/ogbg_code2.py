# This example shows how to top the OGBG leaderboard using GNN+LLM
# hyperparams are hardcoded
import argparse
import gc
import time

import torch
from g_retriever import train

from torch_geometric.datasets import OGBG_Code2
from torch_geometric.nn.models import GAT, GRetriever
from torch_geometric.nn.nlp import LLM

master_prompt = "Please provide the name of the Python function."


def get_loss_ogbg(model, batch, **kwargs) -> torch.Tensor:
    questions = [master_prompt for i in range(len(batch.y))]
    labels = ['|'.join(label) for label in batch.y]
    return model(questions, batch.x.to(torch.float), batch.edge_index,
                 batch.batch, labels, batch.edge_attr, batch.desc)


def inference_step_ogbg(model, batch, **kwargs):
    questions = [master_prompt for i in range(len(batch.y))]
    pred = model.inference(questions, batch.x.to(torch.float),
                           batch.edge_index, batch.batch, batch.edge_attr,
                           batch.desc)
    labels = ['|'.join(label) for label in batch.y]
    eval_data = {
        "pred": pred,
        "question": questions,
        "desc": batch.desc,
        "label": labels
    }
    return eval_data


if __name__ == '__main__':
    gnn_to_use = GAT(in_channels=5, hidden_channels=1024, out_channels=1024,
                     num_layers=4, heads=4)
    # Fits on one GraceHopper
    llm_to_use = LLM(model_name="meta-llama/Meta-Llama-3-8B",
                     num_params=7)
    # This would require a data center scale hardware setup
    # llm_to_use = LLM(model_name="deepseek-ai/DeepSeek-Coder-V2-Base",
    #                  num_params=236)
    CodeRetriever = torch.compile(GRetriever(
        llm=llm_to_use,
        gnn=gnn_to_use,
    ))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--percent_train', type=int, default=100,
        help="Select how much of the training data to use,\
        passing an integer in (0,100]")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    args = parser.parse_args()

    start_time = time.time()
    # TODO, try more epochs with checkpointing on
    train(
        num_epochs=1,
        hidden_channels=None,
        num_gnn_layers=None,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        lr=1e-5,
        checkpointing=False,
        model=CodeRetriever,
        dataset=OGBG_Code2,
        get_loss=get_loss_ogbg,
        inference_step=inference_step_ogbg,
        model_save_name="code_retriever",
        percent_train=args.percent_train,
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    gc.collect()
    print(f"Total Time: {time.time() - start_time:2f}s")
