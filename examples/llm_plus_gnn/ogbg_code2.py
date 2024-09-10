# This example shows how to top the OGBG leaderboard using GNN+LLM
# hyperparams are hardcoded
from torch_geometric.datasets import OGBG_Code2
from g_retriever import train

since = time.time()

def get_loss(model, batch) -> torch.Tensor:
    return model(batch.question, batch.x, batch.edge_index, batch.batch,
                     batch.ptr, '|'.join(batch.label), batch.edge_attr, batch.desc)


def inference_step(model, batch):
	out = model.inference(batch.question, batch.x, batch.edge_index,
                               batch.batch, batch.ptr, batch.edge_attr,
                               batch.desc)
	out["label"] = '|'.join(batch.label)
    return 


prep_time, _, _ = train(
    since, epochs=5, gnn_hidden_channels=1024, num_gnn_layers=4,
    batch_size=16, eval_batch_size=32, lr=1e-5, get_loss=get_loss,
    inference_step=inference_step, checkpointing=True, dataset=OGBG_Code2())
torch.cuda.empty_cache()
torch.cuda.reset_max_memory_allocated()
gc.collect()
e2e_time = round(time.time() - since, 2)
print("E2E time (e2e_time) =", e2e_time, "seconds")
print("E2E tme minus Prep Time =", e2e_time - prep_time, "seconds")
