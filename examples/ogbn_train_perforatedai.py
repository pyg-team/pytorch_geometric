import argparse
import os.path as osp
import time

import psutil
import torch
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset
# Include Perforated AI imports
from perforatedai import pb_globals as PBG
from perforatedai import pb_utils as PBU
from torch import Tensor
from tqdm import tqdm

from torch_geometric import seed_everything
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn.models import GAT, GraphSAGE, SGFormer
from torch_geometric.utils import (
    add_self_loops,
    remove_self_loops,
    to_undirected,
)

# PAI README:
'''
Details about how to get started with Perfroated AI can be found at our home
repository here:
https://github.com/PerforatedAI/PerforatedAI-API

Run docker from torch_geometric directory

   docker run --gpus all -i --shm-size=8g -v .:/pai -w /pai
   -t nvcr.io/nvidia/pyg:24.11-py3 /bin/bash

Within Docker

    pip install -e .
    pip perforatedai
    cd examples

Run original with:

    CUDA_VISIBLE_DEVICES=0 python ogbn_train.py --dataset ogbn-products
    --batch_size 128 --model sage

Results:

    Test Accuracy: 77.06%

Run PAI with:

    CUDA_VISIBLE_DEVICES=0 python ogbn_train_perforatedai.py --dataset
    ogbn-products --batch_size 128 --saveName ogbnPAI --model sage

Results:

    Test Accuracy: 78.05%

'''

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter, )
parser.add_argument(
    '--dataset',
    type=str,
    default='ogbn-arxiv',
    choices=['ogbn-papers100M', 'ogbn-products', 'ogbn-arxiv'],
    help='Dataset name.',
)
parser.add_argument(
    '--dataset_dir',
    type=str,
    default='./data',
    help='Root directory of dataset.',
)
parser.add_argument(
    "--model",
    type=str.lower,
    default='SGFormer',
    choices=['sage', 'gat', 'sgformer'],
    help="Model used for training",
)

parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--num_heads',
                    type=int,
                    default=1,
                    help='number of heads for GAT model.')
parser.add_argument('-b', '--batch_size', type=int, default=1024)
parser.add_argument('--num_workers', type=int, default=12)
parser.add_argument('--max_dendrites', type=int, default=4)
parser.add_argument('--max_dendrite_tries', type=int, default=2)
parser.add_argument('--fan_out',
                    type=int,
                    default=10,
                    help='number of neighbors in each layer')
parser.add_argument('--hidden_channels', type=int, default=256)
parser.add_argument('--lr', type=float, default=0.003)
parser.add_argument('--wd', type=float, default=0.0)
parser.add_argument('--dropout', type=float, default=0.5)
'''
How to name the PAI output files, must be changed to run more than
one test at the same time
'''
parser.add_argument('--saveName', type=str)
parser.add_argument(
    '--use_directed_graph',
    action='store_true',
    help='Whether or not to use directed graph',
)
parser.add_argument(
    '--add_self_loop',
    action='store_true',
    help='Whether or not to add self loop',
)
args = parser.parse_args()

# Set Perforated Backpropagation settings for this training run
'''
# When to switch between Dendrite learning and neuron learning is determined
by when the history of validation score or correlation scores shows those
scores are no longer improving
'''
PBG.switchMode = PBG.doingHistory
# When calculating that just look at the current score, not a recent average
PBG.historyLookback = 1
# How many normal epochs to wait for before switching modes.
PBG.nEpochsToSwitch = 25
PBG.pEpochsToSwitch = 15  # Same as above for Dendrite epochs
'''
The default shape of input tensors will be [batch size, number of neurons
in the layer]
'''
PBG.inputDimensions = [-1, 0]
'''
This allows Dendrites to train as long as they keep improving rather
than capping Dendrite training cycles to only be as long as the first
neuron training cycle.
'''
PBG.capAtN = False
# Stop the run after 4 dendrites are created
PBG.maxDendrites = args.max_dendrites
# If a set of Dendrites does not improve try 2 times before giving up
PBG.maxDendriteTries = args.max_dendrite_tries
'''
Make sure correlation scores improve from epoch to epoch by at least 25% and
a raw value of 1e-4 to conclude that the correlation scores have gone up.
'''
PBG.pbImprovementThreshold = 0.25
PBG.pbImprovementThresholdRaw = 1e-4
'''
This setting defaults to True to ensure that adding dendrites has been
tested before starting a full training run.
'''
PBG.testingDendriteCapacity = False
'''
Weight decay tends to negatively impact Dendrite learning.  This turns off
the warning to consider trying an optimizer without it.
'''
PBG.weightDecayAccepted = True

wall_clock_start = time.perf_counter()

if (args.dataset == 'ogbn-papers100M'
        and (psutil.virtual_memory().total / (1024**3)) < 390):
    print('Warning: may not have enough RAM to run this example.')
    print('Consider upgrading RAM if an error occurs.')
    print('Estimated RAM Needed: ~390GB.')

print(f'Training {args.dataset} with {args.model} model.')

seed_everything(123)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_layers = args.num_layers
num_workers = args.num_workers
num_hidden_channels = args.hidden_channels
batch_size = args.batch_size
root = osp.join(args.dataset_dir, args.dataset)
print('The root is: ', root)
dataset = PygNodePropPredDataset(name=args.dataset, root=root)
split_idx = dataset.get_idx_split()
data = dataset[0]

if not args.use_directed_graph:
    data.edge_index = to_undirected(data.edge_index, reduce='mean')
if args.add_self_loop:
    data.edge_index, _ = remove_self_loops(data.edge_index)
    data.edge_index, _ = add_self_loops(data.edge_index,
                                        num_nodes=data.num_nodes)

data.to(device, 'x', 'y')

train_loader = NeighborLoader(
    data,
    input_nodes=split_idx['train'],
    num_neighbors=[args.fan_out] * num_layers,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    persistent_workers=True,
    disjoint=args.model == "sgformer",
)
val_loader = NeighborLoader(
    data,
    input_nodes=split_idx['valid'],
    num_neighbors=[args.fan_out] * num_layers,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    persistent_workers=True,
    disjoint=args.model == "sgformer",
)
test_loader = NeighborLoader(
    data,
    input_nodes=split_idx['test'],
    num_neighbors=[args.fan_out] * num_layers,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    persistent_workers=True,
    disjoint=args.model == "sgformer",
)


def train(epoch: int) -> tuple[Tensor, float]:
    model.train()

    pbar = tqdm(total=split_idx['train'].size(0))
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = 0
    for batch in train_loader:
        optimizer.zero_grad()
        if args.model == "sgformer":
            out = model(batch.x, batch.edge_index.to(device),
                        batch.batch.to(device))[:batch.batch_size]
        else:
            out = model(batch.x,
                        batch.edge_index.to(device))[:batch.batch_size]
        y = batch.y[:batch.batch_size].squeeze().to(torch.long)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(y).sum())
        pbar.update(batch.batch_size)

    pbar.close()
    loss = total_loss / len(train_loader)
    approx_acc = total_correct / split_idx['train'].size(0)
    return loss, approx_acc


@torch.no_grad()
def test(loader: NeighborLoader) -> float:
    model.eval()

    total_correct = total_examples = 0
    for batch in loader:
        batch = batch.to(device)
        batch_size = batch.num_sampled_nodes[0]
        if args.model == "sgformer":
            out = model(batch.x, batch.edge_index,
                        batch.batch)[:batch.batch_size]
        else:
            out = model(batch.x, batch.edge_index)[:batch_size]
        pred = out.argmax(dim=-1)
        y = batch.y[:batch_size].view(-1).to(torch.long)

        total_correct += int((pred == y).sum())
        total_examples += y.size(0)

    return total_correct / total_examples


def get_model(model_name: str) -> torch.nn.Module:
    if model_name == 'gat':
        model = GAT(
            in_channels=dataset.num_features,
            hidden_channels=num_hidden_channels,
            num_layers=num_layers,
            out_channels=dataset.num_classes,
            dropout=args.dropout,
            heads=args.num_heads,
        )
    elif model_name == 'sage':
        model = GraphSAGE(
            in_channels=dataset.num_features,
            hidden_channels=num_hidden_channels,
            num_layers=num_layers,
            out_channels=dataset.num_classes,
            dropout=args.dropout,
        )
    elif model_name == 'sgformer':
        model = SGFormer(
            in_channels=dataset.num_features,
            hidden_channels=num_hidden_channels,
            out_channels=dataset.num_classes,
            trans_num_heads=args.num_heads,
            trans_dropout=args.dropout,
            gnn_num_layers=num_layers,
            gnn_dropout=args.dropout,
        )
    else:
        raise ValueError(f'Unsupported model type: {model_name}')

    return model


model = get_model(args.model).to(device)
model.reset_parameters()

# Set SAGEConv to be a module to create Dendrite versions of
PBG.moduleNamesToConvert.append('SAGEConv')
# And the modules for SGFormer
PBG.moduleNamesToConvert.append('SGModule')
PBG.moduleNamesToConvert.append('GraphModule')

model = model.to(device)
# Moing this to be called before convertnetwork since it changes pointers
model.reset_parameters()

# This is the main PAI function that converts everything under the hood
# to allow for the addition of dendrites
model = PBU.initializePB(model)
'''
# This can be added to pick up where it left off if something crashes.
print('pre loading')
model = PBU.loadSystem(model, args.saveName, 'latest')
print('loaded')
'''

# This change is required since the PBTracker object handles the scheduler
# All values are the same, just bassed in as kwargs
PBG.pbTracker.setOptimizer(torch.optim.Adam)
PBG.pbTracker.setScheduler(torch.optim.lr_scheduler.ReduceLROnPlateau)
schedArgs = {'mode': 'max', 'patience': 5}
optimArgs = {
    'params': model.parameters(),
    'lr': args.lr,
    'weight_decay': args.wd
}
optimizer, scheduler = PBG.pbTracker.setupOptimizer(model, optimArgs,
                                                    schedArgs)

print(f'Total time before training begins took '
      f'{time.perf_counter() - wall_clock_start:.4f}s')
print('Training...')

times = []
train_times = []
inference_times = []
best_val = 0.
'''
# These lines can be used to run full test with a final model file
# Test trained architechture from before adding the first set of dendrites
#model = PBU.loadSystem(model, args.saveName,'best_model_beforeSwitch_0')
# My result: Test Accuracy: 76.93%

# Test final saved best model
model = PBU.loadSystem(model, args.saveName,'best_model')
# My result: Test Accuracy: 78.01%
'''

epoch = 0
while True:
    train_start = time.perf_counter()
    # Retain the train_acc so that it can be graphed
    loss, train_acc = train(epoch)
    train_times.append(time.perf_counter() - train_start)

    inference_start = time.perf_counter()
    val_acc = test(val_loader)
    inference_times.append(time.perf_counter() - inference_start)

    print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, ',
          f'Train Time: {train_times[-1]:.4f}s')
    print(f'Val: {val_acc * 100.0:.2f}%,')

    if val_acc > best_val:
        best_val = val_acc
    times.append(time.perf_counter() - train_start)
    # Add the extra scores to be graphed
    PBG.pbTracker.addExtraScore(train_acc, 'Train Accuracy')
    '''
    Add the validation score
    This is the function that determines when to add new Dendrites for
    Dendrite training and when to connect trained Dendrites to neurons
    for neuron training.
    model - The model to use going forward.  It is the same if there was not
        a chance and it is a pointer to a new model if there was a change
    improved - Whether or not this validation score improved the old one
    restructured - If a restructing did happen
    trainingComplete - if the tracker has determined that this is the final
    model to use
    '''
    outputs = PBG.pbTracker.addValidationScore(val_acc, model)
    model, restructured, trainingComplete = outputs
    # Need to setup GPU settings of the new model
    model = model.to(device)
    # When training is complete break the loop of epochs
    if (trainingComplete):
        break
    elif (restructured):
        '''
        If the network was restructured reinitialize the optimizer
        and scheduler for the new paramters
        '''
        schedArgs = {
            'mode': 'max',
            'patience': 5
        }  # Make sure this is lower than epochs to switch
        optimArgs = {
            'params': model.parameters(),
            'lr': args.lr,
            'weight_decay': args.wd
        }
        optimizer, scheduler = PBG.pbTracker.setupOptimizer(
            model, optimArgs, schedArgs)
    epoch += 1

print(f'Average Epoch Time on training: '
      f'{torch.tensor(train_times).mean():.4f}s')
print(f'Average Epoch Time on inference: '
      f'{torch.tensor(inference_times).mean():.4f}s')
print(f'Average Epoch Time: {torch.tensor(times).mean():.4f}s')
print(f'Median Epoch Time: {torch.tensor(times).median():.4f}s')
print(f'Best Validation Accuracy: {100.0 * best_val:.2f}%')

print('Testing...')
test_final_acc = test(test_loader)
print(f'Test Accuracy: {100.0 * test_final_acc:.2f}%')
print(f'Total Program Runtime: '
      f'{time.perf_counter() - wall_clock_start:.4f}s')
