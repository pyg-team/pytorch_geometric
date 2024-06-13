import torch


def is_xpu_avaliable() -> bool:
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        return True
    try:
        import intel_extension_for_pytorch as ipex
        return ipex.xpu.is_available()
    except ImportError:
        return False
