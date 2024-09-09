# This example shows how to top the OGBG leaderboard using GNN+LLM
# hyperparams are hardcoded
from torch_geometric.datasets import OGBG_Code2
from g_retriever import train

since = time.time()

def get_loss():
	pass


def inference_step():
	pass


prep_time, _, gnn_llm_eval_outs = train(
    since, args.epochs, args.gnn_hidden_channels, args.num_gnn_layers,
    args.batch_size, args.eval_batch_size, args.lr, get_loss,
    inference_step, checkpointing=True, dataset=OGBG_Code2())
torch.cuda.empty_cache()
torch.cuda.reset_max_memory_allocated()
gc.collect()
e2e_time = round(time.time() - since, 2)
print("E2E time (e2e_time) =", e2e_time, "seconds")
print("E2E tme minus Prep Time =", e2e_time - prep_time, "seconds")
