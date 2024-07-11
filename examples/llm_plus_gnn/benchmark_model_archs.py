# %%
from .g_retriever import benchmark_models, get_loss, inference_step
from torch_geometric.datasets import UpdatedWebQSPDataset
from torch_geometric.nn.models import GRetriever, MLP, GAT

# %%
hidden_channels = 1024
num_gnn_layers = 1
lr=1e-5
epochs=2
batch_size=8
eval_batch_size=16

# %%
ds = UpdatedWebQSPDataset('benchmark_archs')

# %%
model_names = []
model_classes = []
model_kwargs = []
model_type = ["GAT", "MLP"]
models = {"GAT": GAT, "MLP": MLP}
num_layers = [1, 2, 4, 8, 16, 32]
for m_type in model_type:
    for n_layer in num_layers:
        model_names.append(f"{m_type}_{n_layer}")
        model_classes.append(GRetriever)
        kwargs = dict(gnn_hidden_channels=hidden_channels, num_gnn_layers=n_layer, gnn_to_use=models[m_type])
        model_kwargs.append(kwargs)

# %%
benchmark_models(model_classes, model_names, model_kwargs, ds, lr, epochs, batch_size, eval_batch_size, get_loss, inference_step, skip_LLMs=False, tiny_llama=True, force=True)

# %%



