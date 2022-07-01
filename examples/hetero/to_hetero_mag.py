import argparse
import os.path as osp
import time

import torch
import torch.nn.functional as F
from torch.nn import ReLU
from tqdm import tqdm
from torch.profiler import profile, ProfilerActivity

import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
from torch_geometric.loader import HGTLoader, NeighborLoader
from torch_geometric.nn import Linear, SAGEConv, Sequential, to_hetero

parser = argparse.ArgumentParser()
parser.add_argument('--use_hgt_loader', action='store_true')
parser.add_argument('--inference', type=bool, default=False)
parser.add_argument('--profile', type=bool, default=False) # Currently support profile in inference
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
profile_sort = "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"

path = osp.join(osp.dirname(osp.realpath(__file__)), '../../data/OGB')
transform = T.ToUndirected(merge=True)
dataset = OGB_MAG(path, preprocess='metapath2vec', transform=transform)

# Already send node features/labels to GPU for faster access during sampling:
data = dataset[0].to(device, 'x', 'y')

train_input_nodes = ('paper', data['paper'].train_mask)
val_input_nodes = ('paper', data['paper'].val_mask)
kwargs = {'batch_size': 1024, 'num_workers': 6, 'persistent_workers': True}

if not args.use_hgt_loader:
    train_loader = NeighborLoader(data, num_neighbors=[10] * 2, shuffle=True,
                                  input_nodes=train_input_nodes, **kwargs)
    val_loader = NeighborLoader(data, num_neighbors=[10] * 2,
                                input_nodes=val_input_nodes, **kwargs)
else:
    train_loader = HGTLoader(data, num_samples=[1024] * 4, shuffle=True,
                             input_nodes=train_input_nodes, **kwargs)
    val_loader = HGTLoader(data, num_samples=[1024] * 4,
                           input_nodes=val_input_nodes, **kwargs)

model = Sequential('x, edge_index', [
    (SAGEConv((-1, -1), 64), 'x, edge_index -> x'),
    ReLU(inplace=True),
    (SAGEConv((-1, -1), 64), 'x, edge_index -> x'),
    ReLU(inplace=True),
    (Linear(-1, dataset.num_classes), 'x -> x'),
])
model = to_hetero(model, data.metadata(), aggr='sum').to(device)

def trace_handler(p):
    output = p.key_averages().table(sort_by=profile_sort)
    print(output)
    import pathlib
    profile_dir = str(pathlib.Path.cwd()) + '/'
    timeline_file = profile_dir + 'timeline-to-hetero-mag' + '.json'
    p.export_chrome_trace(timeline_file)

@torch.no_grad()
def init_params():
    # Initialize lazy parameters via forwarding a single batch to the model:
    batch = next(iter(train_loader))
    batch = batch.to(device, 'edge_index')
    model(batch.x_dict, batch.edge_index_dict)


def train():
    model.train()

    total_examples = total_loss = 0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        batch = batch.to(device, 'edge_index')
        batch_size = batch['paper'].batch_size
        out = model(batch.x_dict, batch.edge_index_dict)['paper'][:batch_size]
        loss = F.cross_entropy(out, batch['paper'].y[:batch_size])
        loss.backward()
        optimizer.step()

        total_examples += batch_size
        total_loss += float(loss) * batch_size

    return total_loss / total_examples


@torch.no_grad()
def test(loader):
    model.eval()

    total_examples = total_correct = 0
    for batch in tqdm(loader):
        batch = batch.to(device, 'edge_index')
        batch_size = batch['paper'].batch_size
        out = model(batch.x_dict, batch.edge_index_dict)['paper'][:batch_size]
        pred = out.argmax(dim=-1)

        total_examples += batch_size
        total_correct += int((pred == batch['paper'].y[:batch_size]).sum())

    return total_correct / total_examples

@torch.no_grad()
def inference(loader):
    model.eval()
    for batch in tqdm(loader):
        batch = batch.to(device, 'edge_index')
        batch_size = batch['paper'].batch_size
        model(batch.x_dict, batch.edge_index_dict)

init_params()  # Initialize parameters.
if not args.inference:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1, 21):
        loss = train()
        val_acc = test(val_loader)
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val: {val_acc:.4f}')
else:
    for epoch in range(1, 21):
        if epoch == 20:
            if args.profile:
                with profile(activities=[
                    ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    on_trace_ready=trace_handler) as p:
                     inference(val_loader)
                     p.step()
            else:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_start = time.time()
                inference(val_loader)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_end = time.time()
                duration = t_end - t_start
                print("End-to-End time: {} s".format(duration), flush=True)
        else:
            inference(val_loader)
