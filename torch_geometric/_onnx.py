import torch

from torch_geometric import is_compiling


def is_in_onnx_export() -> bool:
    r"""Returns :obj:`True` in case :pytorch:`PyTorch` is exporting to ONNX via
    :meth:`torch.onnx.export`.
    """
    if is_compiling():
        return False
    if torch.jit.is_scripting():
        return False
    return torch.onnx.is_in_onnx_export()
