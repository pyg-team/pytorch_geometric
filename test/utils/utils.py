import torch
import torch_scatter
from packaging import version

reductions = ['sum', 'add', 'mean', 'min', 'max']

dtypes = [torch.half, torch.float, torch.double, torch.int, torch.long]
grad_dtypes = [torch.half, torch.float, torch.double]

if version.parse(torch_scatter.__version__) > version.parse("2.0.9"):
    dtypes.append(torch.bfloat16)
    grad_dtypes.append(torch.bfloat16)

devices = [torch.device('cpu')]
if torch.cuda.is_available():
    devices += [torch.device(f'cuda:{torch.cuda.current_device()}')]


def tensor(x, dtype, device):
    return None if x is None else torch.tensor(x, dtype=dtype, device=device)
