from typing import Any

import torch


def is_mps_available() -> bool:
    r"""Returns a bool indicating if MPS is currently available."""
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        try:  # Github CI may not have access to MPS hardware. Confirm:
            torch.empty(1, device='mps')
            return True
        except Exception:
            return False
    return False


def is_xpu_available() -> bool:
    r"""Returns a bool indicating if XPU is currently available."""
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        return True
    try:
        import intel_extension_for_pytorch as ipex
        return ipex.xpu.is_available()
    except ImportError:
        return False


def device(device: Any) -> torch.device:
    r"""Returns a :class:`torch.device`.

    If :obj:`"auto"` is specified, returns the optimal device depending on
    available hardware.
    """
    if device != 'auto':
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if is_mps_available():
        return torch.device('mps')
    if is_xpu_available():
        return torch.device('xpu')
    return torch.device('cpu')
