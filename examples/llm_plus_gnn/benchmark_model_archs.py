"""Used to benchmark the performance of an untuned/fine tuned LLM against 
GRetriever with various architectures and layer depths."""
# %%
from g_retriever import benchmark_models, get_loss, inference_step
import argparse
from torch_geometric.datasets import UpdatedWebQSPDataset
from torch_geometric.nn.models import GAT, MLP, GRetriever
import torch


# %%
parser = argparse.ArgumentParser(description="Benchmarker for GRetriever")
parser.add_argument("--hidden_channels", type=int, default=1024)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--eval_batch_size", type=int, default=16)
parser.add_argument("--tiny_llama", store_true=True)

parser.add_argument("--custom_dataset", store_true=True)
parser.add_argument("--dataset_path", type=str, required=False)
# Default to WebQSP split
parser.add_argument("--num_train", type=int, default=2826)
parser.add_argument("--num_val", type=int, default=246)
parser.add_argument("--num_test", type=int, default=1628)

args = parser.parse_args()

# %%
hidden_channels = args.hidden_channels
lr = args.learning_rate
epochs = args.epochs
batch_size = args.batch_size
eval_batch_size = args.eval_batch_size

# %%
if not args.custom_dataset:
    ds = UpdatedWebQSPDataset('benchmark_archs')
else:
    # We just assume that the size of the dataset accomodates the
    # train/val/test split, because checking may be expensive.
    dataset = torch.load(args.dataset_path)

    class MockDataset:
        """Utility class to patch the fields in WebQSPDataset used by GRetriever."""
        def __init__(self) -> None:
            pass
        
        @property
        def split_idxs(self) -> dict:
            # Imitates the WebQSP split method
            return {
                "train": torch.arange(args.num_train),
                "val": torch.arange(args.num_val) + args.num_train,
                "test": torch.arange(args.num_test) + args.num_train + args.num_val,
            }
        def __getitem__(self, idx: int):
            return dataset[idx]
    ds = MockDataset()

# %%
model_names = []
model_classes = []
model_kwargs = []
model_type = ["GAT", "MLP"]
models = {"GAT": GAT, "MLP": MLP}
num_layers = [1, 4, 16]
for m_type in model_type:
    for n_layer in num_layers:
        model_names.append(f"{m_type}_{n_layer}")
        model_classes.append(GRetriever)
        kwargs = dict(gnn_hidden_channels=hidden_channels,
                      num_gnn_layers=n_layer, gnn_to_use=models[m_type])
        model_kwargs.append(kwargs)

# %%
benchmark_models(model_classes, model_names, model_kwargs, ds, lr, epochs,
                 batch_size, eval_batch_size, get_loss, inference_step,
                 skip_LLMs=False, tiny_llama=True, force=True)

# TODO Argparse options
