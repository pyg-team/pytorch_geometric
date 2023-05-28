import time

import torch
import torch.nn.functional as F
from torch.optim import Adam

from torch_geometric.loader import DataLoader
from torch_geometric.profile import timeit, torch_profile

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run_train(train_dataset, test_dataset, model, epochs, batch_size, lr,
              lr_decay_factor, lr_decay_step_size, weight_decay):
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    for epoch in range(1, epochs + 1):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()

        train(model, optimizer, train_loader, device)
        test_acc = test(model, test_loader, device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()

        print(f'Epoch: {epoch:03d}, Test: {test_acc:.4f}, '
              f'Duration: {t_end - t_start:.2f}')

        if epoch % lr_decay_step_size == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_decay_factor * param_group['lr']


@torch.no_grad()
def run_inference(test_dataset, model, epochs, batch_size, profiling, bf16):
    model = model.to(device)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    if torch.cuda.is_available():
        amp = torch.cuda.amp.autocast(enabled=False)
    else:
        amp = torch.cpu.amp.autocast(enabled=bf16)

    with amp:
        for epoch in range(1, epochs + 1):
            print("Epoch: ", epoch)
            if epoch == epochs:
                with timeit():
                    inference(model, test_loader, device, bf16)
            else:
                inference(model, test_loader, device, bf16)

        if profiling:
            with torch_profile():
                inference(model, test_loader, device, bf16)


def run(train_dataset, test_dataset, model, epochs, batch_size, lr,
        lr_decay_factor, lr_decay_step_size, weight_decay, inference,
        profiling, bf16):
    if not inference:
        run_train(train_dataset, test_dataset, model, epochs, batch_size, lr,
                  lr_decay_factor, lr_decay_step_size, weight_decay)
    else:
        run_inference(test_dataset, model, epochs, batch_size, profiling, bf16)


def train(model, optimizer, train_loader, device):
    model.train()

    for data in train_loader:
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data.pos, data.batch)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def test(model, test_loader, device):
    model.eval()

    correct = 0
    for data in test_loader:
        data = data.to(device)
        pred = model(data.pos, data.batch).max(1)[1]
        correct += pred.eq(data.y).sum().item()
    test_acc = correct / len(test_loader.dataset)

    return test_acc


@torch.no_grad()
def inference(model, test_loader, device, bf16):
    model.eval()
    for data in test_loader:
        data = data.to(device)
        if bf16:
            data.pos = data.pos.to(torch.bfloat16)
            model = model.to(torch.bfloat16)
        model(data.pos, data.batch)
