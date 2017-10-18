import torch


def save(obj, path):
    with open(path, 'wb') as f:
        torch.save(obj, f)
