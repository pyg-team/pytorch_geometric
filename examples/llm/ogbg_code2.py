# This example shows how to top the OGBG leaderboard using GNN+LLM
# hyperparams are hardcoded
import torch
from g_retriever import train

from torch_geometric.datasets import OGBG_Code2
from torch_geometric.nn.models import GAT, GRetriever
from torch_geometric.nn.nlp import LLM

master_prompt = "Please provide the name of the python function."


def get_loss_ogbg(model, batch, **kwargs) -> torch.Tensor:
    print("correct loss")
    questions = [master_prompt for i in range(len(batch.y))]
    return model(questions, batch.x, batch.edge_index, batch.batch, batch.ptr,
                 '|'.join(batch.y), batch.edge_attr, batch.desc)


def inference_step_ogbg(model, batch, **kwargs):
    print("using correct inferencefunc")
    questions = [master_prompt for i in range(len(batch.y))]
    pred = model.inference(questions, batch.x, batch.edge_index, batch.batch,
                           batch.ptr, batch.edge_attr, batch.desc)
    eval_data = {
        "pred": pred,
        "question": batch.question,
        "desc": batch.desc,
        "label": '|'.join(batch.y)
    }
    return eval_data


gnn_to_use = GAT(in_channels=1024, hidden_channels=1024, out_channels=1024,
                 num_layers=4, heads=4)
llm_to_use = LLM(model_name="meta-llama/CodeLlama-7b-Python-hf", num_params=7)
# This would require a data center scale hardware setup
# llm_to_use = LLM(model_name="deepseek-ai/DeepSeek-Coder-V2-Base",
#                  num_params=236)
CodeRetriever = GRetriever(
    llm=llm_to_use,
    gnn=gnn_to_use,
)

train(num_epochs=5, hidden_channels=None, num_gnn_layers=None, batch_size=16,
      eval_batch_size=32, lr=1e-5, checkpointing=True, model=CodeRetriever,
      dataset=OGBG_Code2, get_loss=get_loss_ogbg,
      inference_step=inference_step_ogbg, model_save_name="code_retriever")
torch.cuda.empty_cache()
torch.cuda.reset_max_memory_allocated()
gc.collect()
