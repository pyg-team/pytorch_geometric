import os
import tempfile
import warnings
from typing import Any
from unittest.mock import patch

import pytest
import torch

from torch_geometric import is_in_onnx_export, safe_onnx_export


class SimpleModel(torch.nn.Module):
    """Simple model for testing ONNX export."""
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def test_is_in_onnx_export() -> None:
    """Test is_in_onnx_export function."""
    assert not is_in_onnx_export()


def test_safe_onnx_export_success() -> None:
    """Test successful ONNX export."""
    model = SimpleModel()
    x = torch.randn(3, 4)

    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        try:
            # Test with tuple args
            result = safe_onnx_export(model, (x, ), f.name)
            assert result is True
            assert os.path.exists(f.name)

            # Test with single tensor (should be converted to tuple)
            result = safe_onnx_export(model, x, f.name)
            assert result is True
        finally:
            if os.path.exists(f.name):
                os.unlink(f.name)


def test_safe_onnx_export_with_skip_on_error() -> None:
    """Test safe_onnx_export with skip_on_error=True."""
    model = SimpleModel()
    x = torch.randn(3, 4)

    # Mock torch.onnx.export to raise SerdeError
    with patch('torch.onnx.export') as mock_export:
        mock_export.side_effect = Exception(
            "onnx_ir.serde.SerdeError: allowzero")

        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            try:
                # Should return False instead of raising
                result = safe_onnx_export(model, (x, ), f.name,
                                          skip_on_error=True)
                assert result is False
            finally:
                if os.path.exists(f.name):
                    os.unlink(f.name)


def test_serde_error_patterns() -> None:
    """Test detection of various SerdeError patterns."""
    model = SimpleModel()
    x = torch.randn(3, 4)

    error_patterns = [
        "onnx_ir.serde.SerdeError: allowzero attribute",
        "ValueError: Value out of range: 1", "serialize_model_into failed",
        "serialize_attribute_into failed"
    ]

    for error_msg in error_patterns:
        with patch('torch.onnx.export') as mock_export:
            mock_export.side_effect = Exception(error_msg)

            with tempfile.NamedTemporaryFile(suffix='.onnx',
                                             delete=False) as f:
                try:
                    result = safe_onnx_export(model, (x, ), f.name,
                                              skip_on_error=True)
                    assert result is False
                finally:
                    if os.path.exists(f.name):
                        os.unlink(f.name)


def test_non_serde_error_reraise() -> None:
    """Test that non-SerdeError exceptions are re-raised."""
    model = SimpleModel()
    x = torch.randn(3, 4)

    with patch('torch.onnx.export') as mock_export:
        mock_export.side_effect = ValueError("Some other error")

        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            try:
                with pytest.raises(ValueError, match="Some other error"):
                    safe_onnx_export(model, (x, ), f.name)
            finally:
                if os.path.exists(f.name):
                    os.unlink(f.name)


def test_dynamo_fallback() -> None:
    """Test dynamo=False fallback strategy."""
    model = SimpleModel()
    x = torch.randn(3, 4)

    call_count = 0

    def mock_export_side_effect(*args: Any, **kwargs: Any) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First call fails
            raise Exception("onnx_ir.serde.SerdeError: allowzero")
        elif call_count == 2 and not kwargs.get('dynamo', True):
            # Second call succeeds with dynamo=False
            return None
        else:
            raise Exception("Unexpected call")

    with patch('torch.onnx.export', side_effect=mock_export_side_effect):
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            try:
                result = safe_onnx_export(model, (x, ), f.name, dynamo=True)
                assert result is True
                assert call_count == 2
            finally:
                if os.path.exists(f.name):
                    os.unlink(f.name)


def test_opset_fallback() -> None:
    """Test opset version fallback strategy."""
    model = SimpleModel()
    x = torch.randn(3, 4)

    call_count = 0

    def mock_export_side_effect(*args: Any, **kwargs: Any) -> None:
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            # First two calls fail (original + dynamo fallback)
            raise Exception("onnx_ir.serde.SerdeError: allowzero")
        elif kwargs.get('opset_version') == 17:
            # Third call succeeds with opset_version=17
            return None
        else:
            raise Exception("Unexpected call")

    with patch('torch.onnx.export', side_effect=mock_export_side_effect):
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            try:
                result = safe_onnx_export(model, (x, ), f.name,
                                          opset_version=18)
                assert result is True
                assert call_count == 3
            finally:
                if os.path.exists(f.name):
                    os.unlink(f.name)


def test_all_strategies_fail() -> None:
    """Test when all workaround strategies fail."""
    model = SimpleModel()
    x = torch.randn(3, 4)

    with patch('torch.onnx.export') as mock_export:
        mock_export.side_effect = Exception(
            "onnx_ir.serde.SerdeError: allowzero")

        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            try:
                # Should raise RuntimeError when skip_on_error=False
                with pytest.raises(RuntimeError,
                                   match="Failed to export model to ONNX"):
                    safe_onnx_export(model, (x, ), f.name, skip_on_error=False)

                # Should return False when skip_on_error=True
                result = safe_onnx_export(model, (x, ), f.name,
                                          skip_on_error=True)
                assert result is False
            finally:
                if os.path.exists(f.name):
                    os.unlink(f.name)


def test_pytest_environment_detection() -> None:
    """Test pytest environment detection for better error messages."""
    model = SimpleModel()
    x = torch.randn(3, 4)

    with patch('torch.onnx.export') as mock_export:
        mock_export.side_effect = Exception(
            "onnx_ir.serde.SerdeError: allowzero")

        # Set pytest environment variable
        with patch.dict(os.environ, {'PYTEST_CURRENT_TEST': 'test_something'}):
            with tempfile.NamedTemporaryFile(suffix='.onnx',
                                             delete=False) as f:
                try:
                    with pytest.raises(RuntimeError) as exc_info:
                        safe_onnx_export(model, (x, ), f.name,
                                         skip_on_error=False)

                    # Should contain pytest-specific guidance
                    assert "pytest environments" in str(exc_info.value)
                    assert "torch.jit.script()" in str(exc_info.value)
                finally:
                    if os.path.exists(f.name):
                        os.unlink(f.name)


def test_warnings_emitted() -> None:
    """Test that appropriate warnings are emitted during workarounds."""
    model = SimpleModel()
    x = torch.randn(3, 4)

    call_count = 0

    def mock_export_side_effect(*args: Any, **kwargs: Any) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise Exception("onnx_ir.serde.SerdeError: allowzero")
        elif call_count == 2:
            return None  # Success on dynamo fallback
        else:
            raise Exception("Unexpected call")

    with patch('torch.onnx.export', side_effect=mock_export_side_effect):
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    result = safe_onnx_export(model, (x, ), f.name,
                                              dynamo=True)

                    assert result is True
                    assert len(w) >= 2  # Initial error + dynamo fallback
                    assert any("allowzero boolean attribute bug" in str(
                        warning.message) for warning in w)
                    assert any(
                        "dynamo=False as workaround" in str(warning.message)
                        for warning in w)
            finally:
                if os.path.exists(f.name):
                    os.unlink(f.name)


@pytest.mark.parametrize(
    "args_input",
    [
        torch.randn(3, 4),  # Single tensor
        (torch.randn(3, 4), ),  # Tuple with one tensor
        (torch.randn(3, 4), torch.randn(3, 2)),  # Tuple with multiple tensors
    ])
def test_args_conversion(args_input: Any) -> None:
    """Test that args are properly converted to tuple format."""
    model = SimpleModel()

    with patch('torch.onnx.export') as mock_export:
        mock_export.return_value = None

        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            try:
                result = safe_onnx_export(model, args_input, f.name)
                assert result is True

                # Check that torch.onnx.export was called with tuple args
                mock_export.assert_called_once()
                call_args = mock_export.call_args[0]
                assert isinstance(call_args[1], tuple)  # args should be tuple
            finally:
                if os.path.exists(f.name):
                    os.unlink(f.name)
