"""Used to benchmark the performance of an untuned/fine tuned LLM against
GRetriever with various architectures and layer depths.
"""
# %%
import argparse
import sys

import torch

from torch_geometric.datasets import WebQSPDataset
from torch_geometric.nn.models import GAT, MLP, GRetriever

sys.path.append('..')
from minimal_demo import (  # noqa: E402 # isort:skip
    benchmark_models, get_loss, inference_step,
)

# %%
parser = argparse.ArgumentParser(
    description="""Benchmarker for GRetriever\n""" +
    """NOTE: Evaluating with smaller samples may result in poorer""" +
    """ performance for the trained models compared to """ +
    """untrained models.""")
parser.add_argument("--hidden_channels", type=int, default=1024)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--eval_batch_size", type=int, default=16)
parser.add_argument("--tiny_llama", action='store_true')

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
if not args.dataset_path:
    ds = WebQSPDataset('benchmark_archs', verbose=True, force_reload=True)
else:
    # We just assume that the size of the dataset accommodates the
    # train/val/test split, because checking may be expensive.
    dataset = torch.load(args.dataset_path)

    class MockDataset:
        """Utility class to patch the fields in WebQSPDataset used by
        GRetriever.
        """
        def __init__(self) -> None:
            pass

        @property
        def split_idxs(self) -> dict:
            # Imitates the WebQSP split method
            return {
                "train":
                torch.arange(args.num_train),
                "val":
                torch.arange(args.num_val) + args.num_train,
                "test":
                torch.arange(args.num_test) + args.num_train + args.num_val,
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
# Use to vary the depth of the GNN model
num_layers = [4]
# Use to vary the number of LLM tokens reserved for GNN output
num_tokens = [1]
for m_type in model_type:
    for n_layer in num_layers:
        for n_tokens in num_tokens:
            model_names.append(f"{m_type}_{n_layer}_{n_tokens}")
            model_classes.append(GRetriever)
            kwargs = dict(gnn_hidden_channels=hidden_channels,
                          num_gnn_layers=n_layer, gnn_to_use=models[m_type],
                          mlp_out_tokens=n_tokens)
            if args.tiny_llama:
                kwargs['llm_to_use'] = 'TinyLlama/TinyLlama-1.1B-Chat-v0.1'
                kwargs['mlp_out_dim'] = 2048
                kwargs['num_llm_params'] = 1
            model_kwargs.append(kwargs)

# %%
benchmark_models(model_classes, model_names, model_kwargs, ds, lr, epochs,
                 batch_size, eval_batch_size, get_loss, inference_step,
                 skip_LLMs=False, tiny_llama=args.tiny_llama, force=True)
