import random
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import torch
from ogb.linkproppred import Evaluator, PygLinkPropPredDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from torch_geometric.nn.models import LPFormer

parser = ArgumentParser()
parser.add_argument('--data_name', type=str, default='ogbl-ppa')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--runs', help="# random seeds to run over", type=int,
                    default=5)
parser.add_argument('--batch_size', type=int, default=32768)
parser.add_argument('--hidden_channels', type=int, default=64)
parser.add_argument('--gnn_layers', type=int, default=3)
parser.add_argument('--dropout', help="Applies to GNN and Transformer",
                    type=float, default=0.1)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--eps', help="PPR precision", type=float, default=5e-5)
parser.add_argument('--thresholds',
                    help="List of cn, 1-hop, >1-hop (in that order)",
                    nargs="+", default=[0, 1e-4, 1e-2])
args = parser.parse_args()

device = torch.device(args.device)

dataset = PygLinkPropPredDataset(name=args.data_name)
data = dataset[0].to(device)
data.edge_index = data.edge_index.to(device)

if hasattr(data, 'x') and data.x is not None:
    data.x = data.x.to(device).to(torch.float)

split_edge = dataset.get_edge_split()
split_data = {
    "train_pos": split_edge['train']['edge'].to(device),
    "valid_pos": split_edge['valid']['edge'].to(device),
    "valid_neg": split_edge['valid']['edge_neg'].to(device),
    "test_pos": split_edge['test']['edge'].to(device),
    "test_neg": split_edge['test']['edge_neg'].to(device)
}

if hasattr(data, 'edge_weight') and data.edge_weight is not None:
    edge_weight = data.edge_weight.to(torch.float)
    data.edge_weight = data.edge_weight.view(-1).to(torch.float)
else:
    edge_weight = torch.ones(data.edge_index.size(1)).to(device).float()

# Convert edge_index to SparseTensor for efficiency
# adj_prop = SparseTensor.from_edge_index(
#     data.edge_index, edge_weight.squeeze(-1),
#     [data.num_nodes, data.num_nodes]).to(device)
adj_prop = torch.sparse_coo_tensor(data.edge_index, edge_weight.squeeze(-1),
                                   [data.num_nodes, data.num_nodes]).to(device)

evaluator_hit = Evaluator(name=args.data_name)

model = LPFormer(data.x.size(-1), args.hidden_channels,
                 num_gnn_layers=args.gnn_layers,
                 ppr_thresholds=args.thresholds, gnn_dropout=args.dropout,
                 transformer_dropout=args.dropout, gcn_cache=True).to(device)

# Get PPR matrix in sparse format
ppr_matrix = model.calc_sparse_ppr(data.edge_index, data.num_nodes,
                                   eps=args.eps).to(device)


def train_epoch():
    model.train()
    train_pos = split_data['train_pos'].to(device)
    adjt_mask = torch.ones(train_pos.size(0), dtype=torch.bool, device=device)

    total_loss = total_examples = 0
    d = DataLoader(range(train_pos.size(0)), args.batch_size, shuffle=True)

    for perm in tqdm(d, "Epoch"):
        edges = train_pos[perm].t()

        # Mask positive input samples - Common strategy during training
        adjt_mask[perm] = 0
        edge2keep = train_pos[adjt_mask, :].t()
        # masked_adj_prop = SparseTensor.from_edge_index(
        #     edge2keep.t(), sparse_sizes=(data['num_nodes'],
        #                                  data['num_nodes'])).to_device(device)
        # masked_adj_prop = masked_adj_prop.to_symmetric()

        # Ensure symmetric
        edge2keep = torch.cat((edge2keep, edge2keep[[1, 0]]), dim=1)
        masked_adj_prop = torch.sparse_coo_tensor(
            edge2keep,
            torch.ones(edge2keep.size(1)).to(device),
            (data['num_nodes'], data['num_nodes'])).to(device)

        # For next batch
        adjt_mask[perm] = 1

        pos_out = model(edges, data.x, masked_adj_prop, ppr_matrix)
        pos_loss = -torch.log(torch.sigmoid(pos_out) + 1e-6).mean()

        # Trivial random sampling
        neg_edges = torch.randint(0, data['num_nodes'],
                                  (edges.size(0), edges.size(1)),
                                  dtype=torch.long, device=edges.device)

        neg_out = model(neg_edges, data.x, adj_prop, ppr_matrix)
        neg_loss = -torch.log(1 - torch.sigmoid(neg_out) + 1e-6).mean()

        loss = pos_loss + neg_loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test():
    # NOTE: Eval for ogbl-citation2 is different
    # See `train.py` in https://github.com/HarryShomer/LPFormer/ for more
    # Also see there for how to eval under the HeaRT setting
    # HeaRT = https://arxiv.org/abs/2306.10453
    model.eval()
    all_preds = defaultdict(list)

    for split_key, split_vals in split_data.items():
        if "train" not in split_key:
            preds = []
            for perm in DataLoader(range(split_vals.size(0)), args.batch_size):
                edges = split_vals[perm].t()
                perm_logits = model(edges, data.x, adj_prop, ppr_matrix)
                preds += [torch.sigmoid(perm_logits).cpu()]

            all_preds[split_key] = torch.cat(preds, dim=0)

    val_hits = evaluator_hit.eval({
        'y_pred_pos': all_preds['valid_pos'],
        'y_pred_neg': all_preds['valid_neg']
    })[f'hits@{evaluator_hit.K}']
    test_hits = evaluator_hit.eval({
        'y_pred_pos': all_preds['test_pos'],
        'y_pred_neg': all_preds['test_neg']
    })[f'hits@{evaluator_hit.K}']

    return val_hits, test_hits


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


# Train over args.runs seeds and average results
# Best result for reach run chosen via validation
val_perf_runs = []
test_perf_runs = []
for run in range(1, args.runs + 1):
    print("=" * 75)
    print(f"RUNNING run={run}")
    print("=" * 75)

    set_seeds(run)
    model.reset_parameters()
    optimizer = torch.optim.Adam(list(model.parameters()), lr=args.lr)

    best_valid = 0
    best_valid_test = 0

    for epoch in range(1, 1 + args.epochs):
        loss = train_epoch()
        print(f"Epoch {epoch} Loss: {loss:.4f}\n")

        if epoch % 5 == 0:
            print("Evaluating model...\n", flush=True)
            eval_val, eval_test = test()

            print(f"Valid Hits@{evaluator_hit.K} = {eval_val}")
            print(f"Test Hits@{evaluator_hit.K} = {eval_test}")

            if eval_val > best_valid:
                best_valid = eval_val
                best_valid_test = eval_test

    print(
        f"\nBest Performance:\n  Valid={best_valid}\n  Test={best_valid_test}")
    val_perf_runs.append(best_valid)
    test_perf_runs.append(best_valid_test)

if args.runs > 1:
    print("\n\n")
    print(f"Results over {args.runs} runs:")
    print(f"  Valid = {np.mean(val_perf_runs)} +/- {np.std(val_perf_runs)}")
    print(f"  Test = {np.mean(test_perf_runs)} +/- {np.std(test_perf_runs)}")
