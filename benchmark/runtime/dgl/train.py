import time

import torch
import torch.nn.functional as F


def train_runtime(model, x, y, mask, epochs, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model = model.to(device)
    x, y, mask = x.to(device), y.to(device), mask.to(device)
    model.train()
    y = y[mask]

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(x)
        loss = F.nll_loss(out[mask], y)
        loss.backward()
        optimizer.step()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_end = time.perf_counter()

    return t_end - t_start
