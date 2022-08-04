import torch

reductions = ['sum', 'mul', 'mean', 'min', 'max']
devices = [torch.device('cpu')]
if torch.cuda.is_available():
    devices += [torch.device(f'cuda:{torch.cuda.current_device()}')]


def tensor(x, device, dtype=torch.float):
    return None if x is None else torch.tensor(x, device=device).to(dtype)
