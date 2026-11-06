"""Minimal example showing how to use ``from_relbench`` output with GRetriever.

This script loads Formula 1 data from RelBench, sanitizes numeric node
features, projects heterogeneous node types into a shared latent space,
converts the graph to homogeneous format, and passes it to GRetriever.

The goal is to demonstrate the projection-first pattern required before
calling ``to_homogeneous()`` on RelBench-derived graphs.

Requirements:
    ``pip install relbench "transformers>=4.51,<5.0" sentencepiece
    accelerate``

Usage:
    ``python relbench_gretriever.py``
    ``python relbench_gretriever.py --epochs 10 --llm Qwen/Qwen2-0.5B``
"""
import argparse

import torch
import torch.nn as nn
from packaging.version import Version
from relbench.datasets import get_dataset

from torch_geometric.contrib.utils import from_relbench
from torch_geometric.data import HeteroData
from torch_geometric.llm.models import LLM, GRetriever
from torch_geometric.nn import GAT, HeteroDictLinear

try:
    import transformers
except ImportError as exc:
    raise RuntimeError(
        'The `transformers` package is required. Install it with: '
        '`pip install "transformers>=4.51,<5.0"`.') from exc

if Version(transformers.__version__) >= Version('5.0'):
    raise RuntimeError(
        f'Unsupported transformers version {transformers.__version__}. '
        'This example requires transformers 4.x. Install with: '
        '`pip install "transformers>=4.51,<5.0"`.')

# CLI options
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='rel-f1',
                    help='RelBench dataset name (default: rel-f1)')
parser.add_argument('--llm', type=str, default='Qwen/Qwen2-0.5B',
                    help='HuggingFace LLM model name')
parser.add_argument('--hidden', type=int, default=64,
                    help='Common projection + GNN hidden dim')
parser.add_argument('--gnn_layers', type=int, default=2,
                    help='Number of GAT layers')
parser.add_argument('--epochs', type=int, default=5, help='Training epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--dtype', type=str, default='bfloat16',
                    choices=['float32', 'bfloat16', 'float16'],
                    help='LLM dtype (use float32 for CPU-only)')
parser.add_argument('--n_gpus', type=int, default=torch.cuda.device_count(),
                    help='Number of GPUs for the LLM (0 for CPU)')
args = parser.parse_args()

dtype_map = {
    'float32': torch.float32,
    'bfloat16': torch.bfloat16,
    'float16': torch.float16,
}
args.torch_dtype = dtype_map[args.dtype]

# Load and sanitize RelBench data
print(f'Loading RelBench {args.dataset} dataset...')
dataset = get_dataset(args.dataset)
db = dataset.get_db()
data = from_relbench(db)

# Replace SQL NULLs with zeros and normalize numeric features.
for node_type in data.node_types:
    if hasattr(data[node_type], 'x') and data[node_type].x is not None:
        x = data[node_type].x
        x = torch.nan_to_num(x, nan=0.0)
        std, mean = torch.std_mean(x, dim=0)
        std = torch.where(std == 0, torch.ones_like(std), std)
        data[node_type].x = (x - mean) / std

print(f'Graph: {len(data.node_types)} node types, '
      f'{len(data.edge_types)} edge types')


# Define the projector for heterogeneous node features
class HeteroFeatureProjector(nn.Module):
    """Projects heterogeneous node features to a common dimension.

    Uses ``HeteroDictLinear`` for node types with numeric features
    and ``nn.Embedding`` for featureless structural tables.
    """
    def __init__(self, data: HeteroData, common_dim: int) -> None:
        super().__init__()
        featured = {}
        self.featureless = []
        for nt in data.node_types:
            x = data[nt].get('x', None)
            if x is not None and x.shape[1] > 0:
                featured[nt] = x.shape[1]
            else:
                self.featureless.append(nt)

        self.lin = HeteroDictLinear(featured, common_dim)
        self.embs = nn.ModuleDict({
            nt:
            nn.Embedding(data[nt].num_nodes, common_dim)
            for nt in self.featureless
        })

    def forward(self, data: HeteroData) -> dict[str, torch.Tensor]:
        """Return a dict of projected features, preserving autograd."""
        x_dict = {nt: data[nt].x for nt in self.lin.lins}
        out = self.lin(x_dict)
        res = {}
        for nt in data.node_types:
            if nt in out:
                res[nt] = out[nt]
            else:
                # These learned embeddings are only valid for this graph.
                # They do not generalize to unseen nodes from another graph.
                res[nt] = self.embs[nt].weight
        return res


projector = HeteroFeatureProjector(data, args.hidden)

# Extract the homogeneous graph topology
# The edge structure is static, while node features are recomputed inside
# the training loop so gradients can propagate through the projector.
homo_topology = data.to_homogeneous()
homo_edge_index = homo_topology.edge_index
print(f'Homogeneous: edge_index={list(homo_edge_index.shape)}')

# Build a small set of example questions for the GRetriever demo.
# These are meant to show the relationship between node types and edge data,
# not to model a full retrieval task.
qa_pairs = [
    (
        'Which entity types appear in this Formula 1 graph?',
        'The graph contains node types such as drivers, constructors, '
        'circuits, races, and teams.',
    ),
    (
        'How is the driver-to-race connection represented?',
        'Drivers are linked to races through result and qualifying edges.',
    ),
    (
        'What role do constructors have in the dataset?',
        'Constructors are part of the race entry structure and connect '
        'teams with drivers.',
    ),
    (
        'Why do we project all node types before calling to_homogeneous?',
        'The projection creates a shared embedding space so GRetriever can '
        'process the graph as a single homogeneous tensor.',
    ),
]

# Build the GRetriever model
print(f'\nInitializing GRetriever with LLM={args.llm}...')

gnn = GAT(
    in_channels=args.hidden,
    hidden_channels=args.hidden,
    num_layers=args.gnn_layers,
    out_channels=args.hidden,
)

llm = LLM(
    model_name=args.llm,
    n_gpus=args.n_gpus if args.n_gpus > 0 else None,
    dtype=args.torch_dtype,
    sys_prompt=('You are an expert assistant that answers questions about '
                'Formula 1 data using knowledge graph context. '
                'Give concise, direct answers.'),
)

model = GRetriever(llm=llm, gnn=gnn)

# Move model components to the LLM device
device = model.llm.device
model.gnn = model.gnn.to(device)
projector = projector.to(device)
homo_edge_index = homo_edge_index.to(device)
data = data.to(device)
print(f'Using device: {device}')

# Training loop
# Include projector parameters so the feature embeddings actually learn.
params = [p for p in model.parameters() if p.requires_grad]
params += list(projector.parameters())
optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.05)

context_str = (
    'This Formula 1 knowledge graph includes drivers, constructors, circuits, '
    'races, and teams, with edges representing race results, qualifying, and '
    'entity relationships.')

print(f'\nTraining {args.epochs} epochs on {len(qa_pairs)} samples...')
model.train()
projector.train()

for epoch in range(1, args.epochs + 1):
    total_loss = 0.0

    for q, a in qa_pairs:
        optimizer.zero_grad()

        # Dynamic projection: compute inside the loop so gradients
        # flow back through the projector.
        projected_dict = projector(data)
        # Stack in data.node_types order, then verify the total node count.
        homo_x = torch.cat([projected_dict[nt] for nt in data.node_types],
                           dim=0)
        assert homo_x.size(0) == homo_topology.num_nodes, (
            'Expected projected homo_x to have the same number of nodes as '
            'the homogeneous topology. If this fails, the node ordering '
            'assumption is incorrect.')

        # Single-graph paradigm: all nodes belong to batch index 0.
        # For mini-batched graph training, a Batch object with graph indices
        # would be required instead.
        batch_idx = torch.zeros(homo_x.size(0), dtype=torch.long,
                                device=device)

        loss = model(
            question=[q],
            x=homo_x,
            edge_index=homo_edge_index,
            batch=batch_idx,
            label=[a],
            additional_text_context=[context_str],
        )

        if loss.isnan():
            raise RuntimeError(
                f'NaN loss on question: "{q}". '
                'Check data normalization or reduce learning rate.')

        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 0.1)
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(qa_pairs)
    print(f'Epoch {epoch:02d}: Loss={avg_loss:.4f}')

# Inference demo
print('\nInference:')
model.eval()
projector.eval()

# Compute static features for inference
with torch.no_grad():
    projected_dict = projector(data)
    homo_x = torch.cat([projected_dict[nt] for nt in data.node_types], dim=0)

test_questions = [
    'Which entity types appear in this Formula 1 graph?',
    'Why do we project all node types before calling to_homogeneous?',
]

for test_q in test_questions:
    with torch.no_grad():
        response = model.inference(
            question=[test_q],
            x=homo_x,
            edge_index=homo_edge_index,
            batch=torch.zeros(homo_x.size(0), dtype=torch.long, device=device),
            additional_text_context=[context_str],
            max_out_tokens=64,
        )
    print(f'Q: {test_q}')
    print(f'A: {response[0]}')
    print()
