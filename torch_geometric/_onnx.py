import warnings
from io import BytesIO
from typing import Any, Union

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


def safe_onnx_export(
    model: torch.nn.Module,
    args: Union[torch.Tensor, tuple],
    f: Union[str, BytesIO],
    **kwargs: Any,
) -> None:
    r"""A safe wrapper around :meth:`torch.onnx.export` that handles known
    ONNX serialization issues in PyTorch Geometric.

    This function provides workarounds for the ``onnx_ir.serde.SerdeError``
    with boolean ``allowzero`` attributes that occurs in certain environments.

    Args:
        model (torch.nn.Module): The model to export.
        args (torch.Tensor or tuple): The input arguments for the model.
        f (str or BytesIO): The file path or BytesIO object to save the model.
        **kwargs: Additional arguments passed to :meth:`torch.onnx.export`.

    Example:
        >>> from torch_geometric.nn import SAGEConv
        >>> from torch_geometric import safe_onnx_export
        >>>
        >>> class MyModel(torch.nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.conv = SAGEConv(8, 16)
        ...     def forward(self, x, edge_index):
        ...         return self.conv(x, edge_index)
        >>>
        >>> model = MyModel()
        >>> x = torch.randn(3, 8)
        >>> edge_index = torch.tensor([[0, 1, 2], [1, 0, 2]])
        >>> safe_onnx_export(model, (x, edge_index), 'model.onnx')
    """
    try:
        # First attempt: standard ONNX export
        torch.onnx.export(model, args, f, **kwargs)

    except Exception as e:
        error_str = str(e)
        error_type = type(e).__name__

        # Check for the specific onnx_ir.serde.SerdeError patterns
        is_allowzero_error = (('onnx_ir.serde.SerdeError' in error_str
                               and 'allowzero' in error_str) or
                              'ValueError: Value out of range: 1' in error_str
                              or (error_type == 'SerdeError'
                                  and 'serialize_model_into' in error_str)
                              or (error_type == 'SerdeError'
                                  and 'serialize_attribute_into' in error_str))

        if is_allowzero_error:
            warnings.warn(
                f"Encountered known ONNX serialization issue ({error_type}). "
                "This is likely the allowzero boolean attribute bug. "
                "Attempting workaround...", UserWarning, stacklevel=2)

            # Apply workaround strategies
            _apply_onnx_allowzero_workaround(model, args, f, **kwargs)

        else:
            # Re-raise other errors
            raise


def _apply_onnx_allowzero_workaround(
    model: torch.nn.Module,
    args: Union[torch.Tensor, tuple],
    f: Union[str, BytesIO],
    **kwargs: Any,
) -> None:
    r"""Apply workaround strategies for onnx_ir.serde.SerdeError with allowzero
    attributes.
    """
    # Strategy 1: Try without dynamo if it was enabled
    if kwargs.get('dynamo', False):
        try:
            kwargs_no_dynamo = kwargs.copy()
            kwargs_no_dynamo['dynamo'] = False

            warnings.warn(
                "Retrying ONNX export with dynamo=False as workaround",
                UserWarning, stacklevel=3)

            torch.onnx.export(model, args, f, **kwargs_no_dynamo)
            return

        except Exception:
            pass

    # Strategy 2: Try with different opset versions
    original_opset = kwargs.get('opset_version', 18)
    for opset_version in [17, 16, 15]:
        if opset_version != original_opset:
            try:
                kwargs_opset = kwargs.copy()
                kwargs_opset['opset_version'] = opset_version

                warnings.warn(
                    f"Retrying ONNX export with opset_version={opset_version}",
                    UserWarning, stacklevel=3)

                torch.onnx.export(model, args, f, **kwargs_opset)
                return

            except Exception:
                continue

    # If all strategies fail, provide helpful error message
    raise RuntimeError(
        "Failed to export model to ONNX due to known serialization issue. "
        "This is caused by a bug in the onnx_ir package where boolean "
        "allowzero attributes cannot be serialized. "
        "Workarounds attempted: dynamo=False and different opset versions. "
        "Consider updating to newer versions of onnx, onnxscript, and "
        "onnx_ir, "
        "or export the model outside of pytest environment.")
