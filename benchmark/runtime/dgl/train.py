import time

import torch
import torch.nn.functional as F


def train_runtime(model, data, epochs, device):
    if hasattr(data, 'features'):
        x = torch.tensor(data.features, dtype=torch.float, device=device)
    else:
        x = None
    mask = data.train_mask if hasattr(data, 'train_mask') else data.train_idx
    y = torch.tensor(data.labels, dtype=torch.long, device=device)[mask]

    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        import torch.mps
        torch.mps.synchronize()
    t_start = time.perf_counter()

    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(x)
        loss = F.nll_loss(out[mask], y.view(-1))
        loss.backward()
        optimizer.step()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.synchronize()
    t_end = time.perf_counter()

    return t_end - t_start
