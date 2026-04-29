"""Example demonstrating how to bridge ``from_relbench`` heterogeneous
graphs to GRetriever for graph-augmented question answering.

This example loads the Formula 1 RelBench dataset, sanitizes the data,
projects all node types into a shared latent space (handling featureless
structural tables via learned embeddings), converts to homogeneous format,
and feeds the result into GRetriever.

.. note::
    Calling ``to_homogeneous()`` directly on RelBench data silently
    drops ALL node features (``x=None``) when any table lacks numeric
    columns. This example shows the correct pattern: sanitize, project
    all types to a common dimension, then convert.

.. note::
    Due to a known upstream issue in PyG ``llm.py`` with
    ``transformers >= 5.0``, this example currently requires
    ``transformers 4.x``.
    (``pip install "transformers>=4.51,<5.0"``)

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
from relbench.datasets import get_dataset

from torch_geometric.llm.models import GRetriever, LLM
from torch_geometric.nn import GAT, HeteroDictLinear
from torch_geometric.utils import from_relbench

# ── CLI ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description='RelBench -> GRetriever example.')
parser.add_argument('--dataset', type=str, default='rel-f1',
                    help='RelBench dataset name (default: rel-f1)')
parser.add_argument('--llm', type=str,
                    default='Qwen/Qwen2-0.5B',
                    help='HuggingFace LLM model name')
parser.add_argument('--hidden', type=int, default=64,
                    help='Common projection + GNN hidden dim')
parser.add_argument('--gnn_layers', type=int, default=2,
                    help='Number of GAT layers')
parser.add_argument('--epochs', type=int, default=5,
                    help='Training epochs')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='Learning rate')
parser.add_argument('--dtype', type=str, default='bfloat16',
                    choices=['float32', 'bfloat16', 'float16'],
                    help='LLM dtype (use float32 for CPU-only)')
parser.add_argument('--n_gpus', type=int, default=1,
                    help='Number of GPUs for the LLM (0 for CPU)')
args = parser.parse_args()

_dtype_map = {
    'float32': torch.float32,
    'bfloat16': torch.bfloat16,
    'float16': torch.float16,
}
args.torch_dtype = _dtype_map[args.dtype]

# ── 1. Load & Sanitize RelBench data ─────────────────────────────────
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


# ── 2. Define Trainable Feature Projector ────────────────────────────
class HeteroFeatureProjector(nn.Module):
    """Projects heterogeneous node features to a common dimension.

    Uses ``HeteroDictLinear`` for node types with numeric features
    and ``nn.Embedding`` for featureless structural tables.
    """
    def __init__(self, data, common_dim: int):
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
            nt: nn.Embedding(data[nt].num_nodes, common_dim)
            for nt in self.featureless
        })

    def forward(self, data):
        """Return a dict of projected features, preserving autograd."""
        x_dict = {nt: data[nt].x for nt in self.lin.lins}
        out = self.lin(x_dict)
        res = {}
        for nt in data.node_types:
            if nt in out:
                res[nt] = out[nt]
            else:
                res[nt] = self.embs[nt].weight
        return res


projector = HeteroFeatureProjector(data, args.hidden)

# ── 3. Extract Homogeneous Topology ──────────────────────────────────
# Topology (edge_index) is static, computed once. Node features (homo_x)
# are computed dynamically inside the training loop so that gradients
# flow back through the projector.
homo_topology = data.to_homogeneous()
homo_edge_index = homo_topology.edge_index
print(f'Homogeneous: edge_index={list(homo_edge_index.shape)}')

# ── 4. Create synthetic Q&A pairs ───────────────────────────────────
# These synthetic Q&A pairs are illustrative.
num_drivers = (data['drivers'].num_nodes
               if 'drivers' in data.node_types else '?')
num_constructors = (data['constructors'].num_nodes
                    if 'constructors' in data.node_types else '?')
num_node_types = len(data.node_types)
num_edge_types = len(data.edge_types)

qa_pairs = [
    ('How many drivers are in the dataset?',
     f'There are {num_drivers} drivers in the Formula 1 dataset.'),
    ('How many constructors are in the dataset?',
     f'There are {num_constructors} constructors.'),
    ('How many types of entities are in the graph?',
     f'The graph has {num_node_types} node types and '
     f'{num_edge_types} edge types.'),
    ('What entity types exist in the Formula 1 knowledge graph?',
     f'The entity types include: {", ".join(data.node_types)}.'),
    ('How are drivers connected to races?',
     'Drivers connect to races through results and qualifying entries.'),
    ('What does this knowledge graph represent?',
     'This graph represents Formula 1 racing data including drivers, '
     'teams, circuits, races, and their relationships.'),
]

# ── 5. Build GRetriever model ────────────────────────────────────────
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
    sys_prompt=(
        'You are an expert assistant that answers questions about '
        'Formula 1 data using knowledge graph context. '
        'Give concise, direct answers.'
    ),
)

model = GRetriever(llm=llm, gnn=gnn)
print('Model initialized.')

# Move model components to the LLM device
device = model.llm.device
model.gnn = model.gnn.to(device)
projector = projector.to(device)
homo_edge_index = homo_edge_index.to(device)
data = data.to(device)
print(f'Using device: {device}')

# ── 6. Training loop ────────────────────────────────────────────────
# Include projector parameters so the feature embeddings actually learn.
params = [p for p in model.parameters() if p.requires_grad]
params += list(projector.parameters())
optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.05)

context_str = (
    f'This is a Formula 1 knowledge graph with {num_node_types} entity '
    f'types ({", ".join(data.node_types)}).'
)

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
        # Stack in data.node_types order (same order as to_homogeneous)
        homo_x = torch.cat(
            [projected_dict[nt] for nt in data.node_types], dim=0)

        # Single-graph paradigm: all nodes belong to batch index 0
        batch_idx = torch.zeros(
            homo_x.size(0), dtype=torch.long, device=device)

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

# ── 7. Inference demo ────────────────────────────────────────────────
print('\nInference:')
model.eval()
projector.eval()

# Compute static features for inference
with torch.no_grad():
    projected_dict = projector(data)
    homo_x = torch.cat(
        [projected_dict[nt] for nt in data.node_types], dim=0)

test_questions = [
    'How many drivers are in this Formula 1 dataset?',
    'What entity types exist in the graph?',
]

for test_q in test_questions:
    with torch.no_grad():
        response = model.inference(
            question=[test_q],
            x=homo_x,
            edge_index=homo_edge_index,
            batch=torch.zeros(homo_x.size(0), dtype=torch.long,
                              device=device),
            additional_text_context=[context_str],
            max_out_tokens=64,
        )
    print(f'Q: {test_q}')
    print(f'A: {response[0]}')
    print()
