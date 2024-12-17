import argparse
import os.path as osp
import time
from typing import Tuple

import psutil
import torch
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from torch import Tensor
from tqdm import tqdm

from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import to_undirected

#Include Perforated AI imports
from perforatedai import globalsFile as gf
from perforatedai import pb_models as PBM
from perforatedai import pb_utils as PBU

'''
PAI README:

Run docker from torch_geometric directory

   docker run --gpus all -i --shm-size=8g -v .:/pai -w /pai -t nvcr.io/nvidia/pyg:24.11-py3 /bin/bash

Within Docker

    pip install -e .
    cd examples
    pip install PAI wheel file
    
Run original with:

    CUDA_VISIBLE_DEVICES=0 python ogbn_train_original.py --dataset ogbn-products --batch_size 128

Results:

    Test Accuracy: 75.52%
    
Run original scheduler with:

    CUDA_VISIBLE_DEVICES=0 python ogbn_train_scheduler.py --dataset ogbn-products --batch_size 128

Results:

    Test Accuracy: 77.51%
    
    
Run PAI with:

    CUDA_VISIBLE_DEVICES=0 python ogbn_trainPAI.py --dataset ogbn-products --batch_size 128 --saveName ogbnPAI
    
Results:

    Test Accuracy: 78.10%
'''


# Set Perforated Backpropagation settings for this training run

# When to switch between Dendrite learning and neuron learning is determined by when the history of validation score or correlation scores shows those scores are no longer improving
gf.switchMode = gf.doingHistory 
# When calculating that just look at the current score, not a recent average
gf.historyLookback = 1
# How many normal epochs to wait for before switching modes.  
gf.nEpochsToSwitch = 25  
gf.pEpochsToSwitch = 15  # Same as above for Dendrite epochs
# The default shape of input tensors will be [batch size, number of neurons in the layer]
gf.inputDimensions = [-1, 0]
# This allows Dendrites to train as long as they keep improving rather than capping
# Dendrite training cycles to only be as long as the first neuron training cycle.
gf.capAtN = False  
# Stop the run after 3 dendrites are created
gf.maxDendrites = 4
# If a set of Dendrites does not improve the system try it 3 times before giving up
gf.maxDendriteTries = 2
# Make sure correlation scores improve from epoch to epoch by at least 25% and a raw value of 1e-4 to conclude that the correlation scores have gone up.
gf.pbImprovementThreshold = 0.25
gf.pbImprovementThresholdRaw = 1e-4

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter, )
parser.add_argument(
    '--dataset',
    type=str,
    default='ogbn-papers100M',
    choices=['ogbn-papers100M', 'ogbn-products'],
    help='Dataset name.',
)
parser.add_argument(
    '--dataset_dir',
    type=str,
    default='./data',
    help='Root directory of dataset.',
)
parser.add_argument(
    "--dataset_subdir",
    type=str,
    default="ogb-papers100M",
    help="directory of dataset.",
)
parser.add_argument(
    '--use_gat',
    action='store_true',
    help='Whether or not to use GAT model',
)
parser.add_argument(
    '--verbose',
    action='store_true',
    help='Whether or not to generate statistical report',
)
parser.add_argument('--device', type=str, default='cuda')
# Perforated Backpropagation will automatically quick when it is time to early stop, so just set epochs to be higher than that takes
parser.add_argument('-e', '--epochs', type=int, default=10000,
                    help='number of training epochs.')
parser.add_argument('--num_layers', type=int, default=3,
                    help='number of layers.')
parser.add_argument('--num_heads', type=int, default=2,
                    help='number of heads for GAT model.')
parser.add_argument('-b', '--batch_size', type=int, default=1024,
                    help='batch size.')
parser.add_argument('--num_workers', type=int, default=12,
                    help='number of workers.')
parser.add_argument('--fan_out', type=int, default=10,
                    help='number of neighbors in each layer')
parser.add_argument('--hidden_channels', type=int, default=256,
                    help='number of hidden channels.')
# This can be set to 0 to run this code without Perforated Backpropagation happening.
parser.add_argument('--doingPB', type=int, default=1,                    
                    help='doing PB')
parser.add_argument('--lr', type=float, default=0.003)
parser.add_argument('--wd', type=float, default=0.0,
                    help='weight decay for the optimizer')
parser.add_argument('--dropout', type=float, default=0.5)
# How to name the PAI output files, must be changed if you run more than one test at a time
parser.add_argument('--saveName', type=str)
parser.add_argument(
    '--use_directed_graph',
    action='store_true',
    help='Whether or not to use directed graph',
)
args = parser.parse_args()
if "papers" in str(args.dataset) and (psutil.virtual_memory().total /
                                      (1024**3)) < 390:
    print("Warning: may not have enough RAM to use this many GPUs.")
    print("Consider upgrading RAM if an error occurs.")
    print("Estimated RAM Needed: ~390GB.")
verbose = args.verbose
if verbose:
    wall_clock_start = time.perf_counter()
    if args.use_gat:
        print(f"Training {args.dataset} with GAT model.")
    else:
        print(f"Training {args.dataset} with GraphSage model.")

if not torch.cuda.is_available():
    args.device = "cpu"
device = torch.device(args.device)

num_epochs = args.epochs
num_layers = args.num_layers
num_workers = args.num_workers
num_hidden_channels = args.hidden_channels
batch_size = args.batch_size

root = osp.join(args.dataset_dir, args.dataset_subdir)
print('The root is: ', root)
dataset = PygNodePropPredDataset(name=args.dataset, root=root)
split_idx = dataset.get_idx_split()
evaluator = Evaluator(name=args.dataset)

data = dataset[0]
if not args.use_directed_graph:
    data.edge_index = to_undirected(data.edge_index, reduce='mean')

data.to(device, 'x', 'y')

train_loader = NeighborLoader(
    data,
    input_nodes=split_idx['train'],
    num_neighbors=[args.fan_out] * num_layers,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    persistent_workers=True,
)
val_loader = NeighborLoader(
    data,
    input_nodes=split_idx['valid'],
    num_neighbors=[args.fan_out] * num_layers,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    persistent_workers=True,
)
test_loader = NeighborLoader(
    data,
    input_nodes=split_idx['test'],
    num_neighbors=[args.fan_out] * num_layers,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    persistent_workers=True,
)


def train(epoch: int) -> Tuple[Tensor, float]:
    model.train()

    pbar = tqdm(total=split_idx['train'].size(0))
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = 0
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index.to(device))[:batch.batch_size]
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
def test(loader: NeighborLoader, val_steps=None) -> float:
    model.eval()

    total_correct = total_examples = 0
    for i, batch in enumerate(loader):
        if val_steps is not None and i >= val_steps:
            break
        batch = batch.to(device)
        batch_size = batch.num_sampled_nodes[0]
        out = model(batch.x, batch.edge_index)[:batch_size]
        pred = out.argmax(dim=-1)
        y = batch.y[:batch_size].view(-1).to(torch.long)

        total_correct += int((pred == y).sum())
        total_examples += y.size(0)

    return total_correct / total_examples


if args.use_gat:
    from torch_geometric.nn.models import GAT
    model = GAT(
        in_channels=dataset.num_features,
        hidden_channels=num_hidden_channels,
        num_layers=num_layers,
        out_channels=dataset.num_classes,
        dropout=args.dropout,
        heads=args.num_heads,
    )
else:
    from torch_geometric.nn.models import GraphSAGE
    model = GraphSAGE(
        in_channels=dataset.num_features,
        hidden_channels=num_hidden_channels,
        num_layers=num_layers,
        out_channels=dataset.num_classes,
        dropout=args.dropout,
    )

# Set SAGEConv to be a module to create Dendrite versions of
gf.moduleNamesToConvert.append('SAGEConv')
model = model.to(device)
# Moing this to be called before convert network since it changes some pointers
model.reset_parameters()

# This is the main PAI function that converts everything under the hood
# to allow for the addition of dendrites
model = PBU.convertNetwork(model)

# This initializes the Perforated Backpropagation Tracker object which organizes
# communication between each individual Dendrite convereted module within a full network
gf.pbTracker.initialize(
    doingPB = args.doingPB, #This can be set to false if you want to do just normal training 
    saveName=args.saveName,  # Change the save name for different parameter runs
maximizingScore=True, # True for maximizing validation score, false for minimizing validation loss
makingGraphs=True)  # True if you want graphs to be saved

# This change is required since the PBTracker object handles the scheduler
# All values are the same, just bassed in as kwargs
gf.pbTracker.setOptimizer(torch.optim.Adam)
gf.pbTracker.setScheduler(torch.optim.lr_scheduler.ReduceLROnPlateau)
schedArgs = {'mode':'max', 'patience': 5}
optimArgs = {'params':model.parameters(),'lr':args.lr, 'weight_decay':args.wd}
optimizer, scheduler = gf.pbTracker.setupOptimizer(model, optimArgs, schedArgs)


if verbose:
    prep_time = round(time.perf_counter() - wall_clock_start, 2)
    print("Total time before training begins (prep_time)=", prep_time,
          "seconds")
    print("Training...")

test_accs = []
val_accs = []
times = []
train_times = []
inference_times = []
best_val = best_test = 0.
start = time.time()

# Moved above before convert network
#model.reset_parameters()

'''
# These lines can be used to run full test with a final model file
num_epochs = 0
# Test trained architechture from before adding the first set of dendrites
#model = PBU.loadSystem(model, args.saveName,'best_model_beforeSwitch_0')
# My result: Test Accuracy: 76.93%

# Test final saved best model
model = PBU.loadSystem(model, args.saveName,'best_model')
# My result: Test Accuracy: 78.01%

'''

for epoch in range(1, num_epochs + 1):
    train_start = time.time()
    
    # Retain the train_acc so that it can be graphed
    loss, train_acc = train(epoch)
    train_end = time.time()
    train_times.append(train_end - train_start)

    inference_start = time.time()
    '''
    Testing every time is normally good, but this dataset has
    17290 test samples and only 1537 training samples.
    Feels like a waste of compute time to run a full test every epoch that
    is over 10x as long as training.
    To save time this is adjusted so test subsamples during N (normal) training epochs
    And both only run 5 batches during P (Perforated Backpropagation) training epochs
    P batches do not need testing because the weights being adjusted to not impact the
    forward pass of the network.
    '''
    val_count = None
    test_count = 308 # num val samples
    if(gf.pbTracker.memberVars['mode'] == 'p'):
        val_count = 5
        test_count = 5
    val_acc = test(val_loader, val_count)
    test_acc = test(test_loader, test_count)

    inference_times.append(time.time() - inference_start)
    test_accs.append(test_acc)
    val_accs.append(val_acc)
    print(
        f'Epoch {epoch:02d}, Loss: {loss:.4f}, Train Time: {train_end - train_start:.4f}s'
    )
    print(f'Val: {val_acc * 100.0:.2f}%,', f'Test: {test_acc * 100.0:.2f}%')

    if val_acc > best_val:
        best_val = val_acc
        best_test = test_acc
    times.append(time.time() - train_start)
    # Add the extra scores to be graphed
    gf.pbTracker.addExtraScore(train_acc, 'Train Accuracy')
    gf.pbTracker.addTestScore(test_acc, 'Test Accuracy')
    
    '''
    Add the validation score
    This is the function that determines when to add new Dendrites for Dendrite training
    and when to connect trained Dendrites to neurons for neuron training.
    model - The model to use going forward.  It is the same if there was not a chance
        and it is a pointer to a new model if there was a change
    improved - Whether or not this validation score improved the old one
    restructured - If a restructing did happen
    trainingComplete - if the tracker has determined that this is the final model to use
    '''
    model, improved, restructured, trainingComplete = gf.pbTracker.addValidationScore(val_acc, 
    model,
    args.saveName)
    # Need to setup GPU settings of the new model
    model = model.to(device)
    # When training is complete break the loop of epochs
    if(trainingComplete):
        break
    # If the network was restructured reinitialize the optimizer and scheduler for the new paramters
    elif(restructured):
        schedArgs = {'mode':'max', 'patience': 5} #Make sure this is lower than epochs to switch
        optimArgs = {'params':model.parameters(),'lr':args.lr, 'weight_decay':args.wd}
        optimizer, scheduler = gf.pbTracker.setupOptimizer(model, optimArgs, schedArgs)

if verbose:
    test_acc = torch.tensor(test_accs)
    val_acc = torch.tensor(val_accs)
    print('============================')
    print("Average Epoch Time on training: {:.4f}".format(
        torch.tensor(train_times).mean()))
    print("Average Epoch Time on inference: {:.4f}".format(
        torch.tensor(inference_times).mean()))
    print(f"Average Epoch Time: {torch.tensor(times).mean():.4f}")
    print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")
    print(f'Final Test: {test_acc.mean():.4f} ± {test_acc.std():.4f}')
    print(f'Final Validation: {val_acc.mean():.4f} ± {val_acc.std():.4f}')
    print(f"Best validation accuracy: {best_val:.4f}")
    print(f"Best testing accuracy: {best_test:.4f}")

if verbose:
    print("Testing...")
test_final_acc = test(test_loader)
print(f'Test Accuracy: {100.0 * test_final_acc:.2f}%')
if verbose:
    total_time = round(time.perf_counter() - wall_clock_start, 2)
    print("Total Program Runtime (total_time) =", total_time, "seconds")
