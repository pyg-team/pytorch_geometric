# This example shows how to top the OGBG leaderboard using GNN+LLM
# hyperparams are hardcoded
from g_retriever import train

from torch_geometric.datasets import OGBG_Code2
from torch_geometric.nn.models import GRetriever

since = time.time()

def get_loss(model, batch, **kwargs) -> torch.Tensor:
    return model(batch.question, batch.x, batch.edge_index, batch.batch,
                     batch.ptr, '|'.join(batch.label), batch.edge_attr, batch.desc)


def inference_step(model, batch, **kwargs):
	out = model.inference(batch.question, batch.x, batch.edge_index,
                               batch.batch, batch.ptr, batch.edge_attr,
                               batch.desc)
	eval_data = {
                "pred": out,
                "question": batch.question,
                "desc": batch.desc,
                "label": '|'.join(batch.label)
            }
    return eval_data

CodeRetriever = GRetriever(llm_to_use="deepseek-ai/DeepSeek-Coder-V2-Base",
                               gnn_hidden_channels=1024,
                               num_gnn_layers=4)

prep_time, _, _ = train(
    since, epochs=5, gnn_hidden_channels=None, num_gnn_layers=None,
    batch_size=16, eval_batch_size=32, lr=1e-5, get_loss=get_loss,
    inference_step=inference_step, checkpointing=True, model=CodeRetriever, dataset=OGBG_Code2())
torch.cuda.empty_cache()
torch.cuda.reset_max_memory_allocated()
gc.collect()
e2e_time = round(time.time() - since, 2)
print("E2E time (e2e_time) =", e2e_time, "seconds")
print("E2E tme minus Prep Time =", e2e_time - prep_time, "seconds")
