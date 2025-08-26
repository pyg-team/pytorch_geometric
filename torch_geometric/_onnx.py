import warnings
from os import PathLike
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
    args: Union[torch.Tensor, tuple[Any, ...]],
    f: Union[str, PathLike[Any], None],
    skip_on_error: bool = False,
    **kwargs: Any,
) -> bool:
    r"""A safe wrapper around :meth:`torch.onnx.export` that handles known
    ONNX serialization issues in PyTorch Geometric.

    This function provides workarounds for the ``onnx_ir.serde.SerdeError``
    with boolean ``allowzero`` attributes that occurs in certain environments.

    Args:
        model (torch.nn.Module): The model to export.
        args (torch.Tensor or tuple): The input arguments for the model.
        f (str or PathLike): The file path to save the model.
        skip_on_error (bool): If True, return False instead of raising when
            workarounds fail. Useful for CI environments.
        **kwargs: Additional arguments passed to :meth:`torch.onnx.export`.

    Returns:
        bool: True if export succeeded, False if skipped due to known issues
              (only when skip_on_error=True).

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
        >>> success = safe_onnx_export(model, (x, edge_index), 'model.onnx')
        >>>
        >>> # For CI environments:
        >>> success = safe_onnx_export(model, (x, edge_index), 'model.onnx',
        ...                             skip_on_error=True)
        >>> if not success:
        ...     print("ONNX export skipped due to known upstream issue")
    """
    # Convert single tensor to tuple for torch.onnx.export compatibility
    if isinstance(args, torch.Tensor):
        args = (args, )

    try:
        # First attempt: standard ONNX export
        torch.onnx.export(model, args, f, **kwargs)
        return True

    except Exception as e:
        error_str = str(e)
        error_type = type(e).__name__

        # Check for the specific onnx_ir.serde.SerdeError patterns
        is_allowzero_error = (('onnx_ir.serde.SerdeError' in error_str
                               and 'allowzero' in error_str) or
                              'ValueError: Value out of range: 1' in error_str
                              or 'serialize_model_into' in error_str
                              or 'serialize_attribute_into' in error_str)

        if is_allowzero_error:
            warnings.warn(
                f"Encountered known ONNX serialization issue ({error_type}). "
                "This is likely the allowzero boolean attribute bug. "
                "Attempting workaround...", UserWarning, stacklevel=2)

            # Apply workaround strategies
            return _apply_onnx_allowzero_workaround(model, args, f,
                                                    skip_on_error, **kwargs)

        else:
            # Re-raise other errors
            raise


def _apply_onnx_allowzero_workaround(
    model: torch.nn.Module,
    args: tuple[Any, ...],
    f: Union[str, PathLike[Any], None],
    skip_on_error: bool = False,
    **kwargs: Any,
) -> bool:
    r"""Apply workaround strategies for onnx_ir.serde.SerdeError with allowzero
    attributes.

    Returns:
        bool: True if export succeeded, False if skipped (when
              skip_on_error=True).
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
            return True

        except Exception:
            pass

    # Strategy 2: Try with different opset versions
    original_opset = kwargs.get('opset_version', 18)
    for opset_version in [17, 16, 15, 14, 13, 11]:
        if opset_version != original_opset:
            try:
                kwargs_opset = kwargs.copy()
                kwargs_opset['opset_version'] = opset_version

                warnings.warn(
                    f"Retrying ONNX export with opset_version={opset_version}",
                    UserWarning, stacklevel=3)

                torch.onnx.export(model, args, f, **kwargs_opset)
                return True

            except Exception:
                continue

    # Strategy 3: Try legacy export (non-dynamo with older opset)
    try:
        kwargs_legacy = kwargs.copy()
        kwargs_legacy['dynamo'] = False
        kwargs_legacy['opset_version'] = 11

        warnings.warn(
            "Retrying ONNX export with legacy settings "
            "(dynamo=False, opset_version=11)", UserWarning, stacklevel=3)

        torch.onnx.export(model, args, f, **kwargs_legacy)
        return True

    except Exception:
        pass

    # Strategy 4: Try with minimal settings
    try:
        minimal_kwargs: dict[str, Any] = {
            'opset_version': 11,
            'dynamo': False,
        }
        # Add optional parameters if they exist
        if kwargs.get('input_names') is not None:
            minimal_kwargs['input_names'] = kwargs.get('input_names')
        if kwargs.get('output_names') is not None:
            minimal_kwargs['output_names'] = kwargs.get('output_names')

        warnings.warn(
            "Retrying ONNX export with minimal settings as last resort",
            UserWarning, stacklevel=3)

        torch.onnx.export(model, args, f, **minimal_kwargs)
        return True

    except Exception:
        pass

    # If all strategies fail, handle based on skip_on_error flag
    import os
    pytest_detected = 'PYTEST_CURRENT_TEST' in os.environ or 'pytest' in str(f)

    if skip_on_error:
        # For CI environments: skip gracefully instead of failing
        warnings.warn(
            "ONNX export skipped due to known upstream issue "
            "(onnx_ir.serde.SerdeError). "
            "This is caused by a bug in the onnx_ir package where boolean "
            "allowzero attributes cannot be serialized. All workarounds "
            "failed. Consider updating packages: pip install --upgrade onnx "
            "onnxscript "
            "onnx_ir", UserWarning, stacklevel=3)
        return False

    # For regular usage: provide detailed error message
    error_msg = (
        "Failed to export model to ONNX due to known serialization issue. "
        "This is caused by a bug in the onnx_ir package where boolean "
        "allowzero attributes cannot be serialized. "
        "Workarounds attempted: dynamo=False, multiple opset versions, "
        "and legacy export. ")

    if pytest_detected:
        error_msg += (
            "\n\nThis error commonly occurs in pytest environments. "
            "Try one of these solutions:\n"
            "1. Run the export outside of pytest (in a regular Python "
            "script)\n"
            "2. Update packages: pip install --upgrade onnx onnxscript "
            "onnx_ir\n"
            "3. Use torch.jit.script() instead of ONNX export for testing\n"
            "4. Use safe_onnx_export(..., skip_on_error=True) to skip "
            "gracefully in CI")
    else:
        error_msg += ("\n\nTry updating packages: pip install --upgrade onnx "
                      "onnxscript onnx_ir")

    raise RuntimeError(error_msg)
