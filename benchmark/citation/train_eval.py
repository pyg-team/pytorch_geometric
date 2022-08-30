import time

import torch
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam

from torch_geometric.profile import timeit, torch_profile
from torch_geometric.utils import index_to_mask

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


def run_train(dataset, model, runs, epochs, lr, weight_decay, early_stopping,
              permute_masks=None, logger=None):
    val_losses, accs, durations = [], [], []
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


@torch.no_grad()
def run_inference(dataset, model, epochs, profiling, bf16, permute_masks=None,
                  logger=None):
    data = dataset[0]
    if permute_masks is not None:
        data = permute_masks(data, dataset.num_classes)
    data = data.to(device)

    model.to(device).reset_parameters()

    if torch.cuda.is_available():
        amp = torch.cuda.amp.autocast(enabled=False)
    else:
        amp = torch.cpu.amp.autocast(enabled=bf16)
    if bf16:
        data.x = data.x.to(torch.bfloat16)

    with amp:
        for epoch in range(1, epochs + 1):
            if epoch == epochs:
                with timeit():
                    inference(model, data)
            else:
                inference(model, data)

        if profiling:
            with torch_profile():
                inference(model, data)


def run(dataset, model, runs, epochs, lr, weight_decay, early_stopping,
        inference, profiling, bf16, permute_masks=None, logger=None):
    if not inference:
        run_train(dataset, model, runs, epochs, lr, weight_decay,
                  early_stopping, permute_masks, logger)
    else:
        run_inference(dataset, model, epochs, profiling, bf16, permute_masks,
                      logger)


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


@torch.no_grad()
def inference(model, data):
    model.eval()
    model(data)
