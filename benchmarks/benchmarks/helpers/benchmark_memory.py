import torch


def _hook_memory():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        used_memory = torch.cuda.max_memory_allocated()
    else:
        used_memory = np.nan
    return used_memory
