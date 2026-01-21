import argparse
import os.path as osp

import torch
import torch.optim as optim

from torch_geometric.contrib.nn import SimplE
from torch_geometric.datasets import FB15k_237

# Parse command-line arguments for hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--hidden_channels', type=int, default=200,
                    help='Hidden embedding size (default: 200)')
parser.add_argument('--batch_size', type=int, default=1000,
                    help='Batch size (default: 1000)')
parser.add_argument('--lr', type=float, default=0.05,
                    help='Learning rate (default: 0.05)')
parser.add_argument('--epochs', type=int, default=500,
                    help='Number of epochs (default: 500)')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..', 'data',
                'FB15k')

# Load the FB15k-237 dataset splits
# FB15k-237 is a subset of Freebase with 237 relations and 14,951 entities
train_data = FB15k_237(path, split='train')[0].to(device)
val_data = FB15k_237(path, split='val')[0].to(device)
test_data = FB15k_237(path, split='test')[0].to(device)

# Initialize the SimplE model
# SimplE uses two embeddings per entity (head/tail) and two per
# relation (forward/inverse)
model = SimplE(
    num_nodes=train_data.num_nodes,
    num_relations=train_data.num_edge_types,
    hidden_channels=args.hidden_channels,
).to(device)

loader = model.loader(
    head_index=train_data.edge_index[0],
    rel_type=train_data.edge_type,
    tail_index=train_data.edge_index[1],
    batch_size=args.batch_size,
    shuffle=True,
)

# Use Adagrad optimizer as recommended in the SimplE paper
optimizer = optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=1e-6)


def train():
    """Trains the SimplE model for one epoch."""
    model.train()
    total_loss = total_examples = 0
    for head_index, rel_type, tail_index in loader:
        optimizer.zero_grad()
        # Compute loss (includes both positive and negative sampling)
        loss = model.loss(head_index, rel_type, tail_index)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * head_index.numel()
        total_examples += head_index.numel()
    return total_loss / total_examples


@torch.no_grad()
def test(data):
    """Evaluates the model on the given dataset.

    Returns:
        tuple: (mean_rank, mrr, hits_at_k) evaluation metrics
    """
    model.eval()
    return model.test(
        head_index=data.edge_index[0],
        rel_type=data.edge_type,
        tail_index=data.edge_index[1],
        batch_size=20000,
        k=10,
    )


for epoch in range(1, args.epochs + 1):
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    if epoch % 25 == 0:
        rank, mrr, hits = test(val_data)
        print(f'Epoch: {epoch:03d}, Val Mean Rank: {rank:.2f}, '
              f'Val MRR: {mrr:.4f}, Val Hits@10: {hits:.4f}')

rank, mrr, hits_at_10 = test(test_data)
print(f'Test Mean Rank: {rank:.2f}, Test MRR: {mrr:.4f}, '
      f'Test Hits@10: {hits_at_10:.4f}')
