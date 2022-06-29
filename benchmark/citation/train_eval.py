import time

import torch
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam
from torch.profiler import profile, ProfilerActivity

from torch_geometric.utils import index_to_mask

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
profile_sort = "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"


def random_planetoid_splits(data, num_classes):
    # Set new random planetoid splits:
    # * 20 * num_classes labels for training
    # * 500 labels for validation
    # * 1000 labels for testing

    indices = []
    for i in range(num_classes):
        index = (data.y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:20] for i in indices], dim=0)

    rest_index = torch.cat([i[20:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(rest_index[:500], size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index[500:1500], size=data.num_nodes)

    return data

def trace_handler(p):
    output = p.key_averages().table(sort_by=profile_sort)
    print(output)
    import pathlib
    profile_dir = str(pathlib.Path.cwd()) + '/'
    timeline_file = profile_dir + 'timeline' + '.json'
    p.export_chrome_trace(timeline_file)

def run(dataset, model, runs, epochs, lr, weight_decay, early_stopping, inference, profiling,
        permute_masks=None, logger=None):
    val_losses, accs, durations = [], [], []
    if not inference:
        for _ in range(runs):
            data = dataset[0]
            if permute_masks is not None:
                data = permute_masks(data, dataset.num_classes)
            data = data.to(device)

            model.to(device).reset_parameters()
            optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            t_start = time.perf_counter()

            best_val_loss = float('inf')
            test_acc = 0
            val_loss_history = []

            for epoch in range(1, epochs + 1):
                train(model, optimizer, data)
                eval_info = evaluate(model, data)
                eval_info['epoch'] = epoch

                if logger is not None:
                    logger(eval_info)

                if eval_info['val_loss'] < best_val_loss:
                    best_val_loss = eval_info['val_loss']
                    test_acc = eval_info['test_acc']

                val_loss_history.append(eval_info['val_loss'])
                if early_stopping > 0 and epoch > epochs // 2:
                    tmp = tensor(val_loss_history[-(early_stopping + 1):-1])
                    if eval_info['val_loss'] > tmp.mean().item():
                        break

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            t_end = time.perf_counter()

            val_losses.append(best_val_loss)
            accs.append(test_acc)
            durations.append(t_end - t_start)
        loss, acc, duration = tensor(val_losses), tensor(accs), tensor(durations)

        print(f'Val Loss: {float(loss.mean()):.4f}, '
            f'Test Accuracy: {float(acc.mean()):.3f} ± {float(acc.std()):.3f}, '
            f'Duration: {float(duration.mean()):.3f}s')
    else:
        for i in range(runs):
            data = dataset[0]
            if permute_masks is not None:
                data = permute_masks(data, dataset.num_classes)
            data = data.to(device)

            model.to(device).reset_parameters()

            for epoch in range(1, epochs + 1):
                if i == int(runs / 2) and epoch == int(epochs / 2):
                    if profiling:
                        with profile(activities=[
                            ProfilerActivity.CPU, ProfilerActivity.CUDA],
                            on_trace_ready=trace_handler) as p:
                            test(model, data)
                            p.step()
                    else:
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        t_start = time.time()

                        test(model, data)

                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        t_end = time.time()
                        duration = t_end - t_start
                        print("End-to-End time: {} s".format(duration), flush=True)
                else:
                    test(model, data)

def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()


def evaluate(model, data):
    model.eval()

    with torch.no_grad():
        logits = model(data)

    outs = {}
    for key in ['train', 'val', 'test']:
        mask = data[f'{key}_mask']
        loss = F.nll_loss(logits[mask], data.y[mask]).item()
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

        outs[f'{key}_loss'] = loss
        outs[f'{key}_acc'] = acc

    return outs

def test(model, data):
    model.eval()
    with torch.no_grad():
        model(data)
