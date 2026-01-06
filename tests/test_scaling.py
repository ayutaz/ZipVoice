"""Tests for the logaddexp_onnx function in scaling.py.

This tests that our Sentis-compatible log(1+x) implementation
produces the same results as torch.logaddexp.
"""

import math

import pytest
import torch

from zipvoice.models.modules.scaling import logaddexp_onnx


class TestLogaddexpOnnx:
    """Test suite for logaddexp_onnx function."""

    def test_basic_values(self):
        """Test that logaddexp_onnx produces correct results for basic inputs."""
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([1.0, 2.0, 3.0])

        result = logaddexp_onnx(x, y)
        expected = torch.logaddexp(x, y)

        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_different_values(self):
        """Test with different x and y values."""
        x = torch.tensor([0.0, 1.0, -1.0, 10.0])
        y = torch.tensor([1.0, 0.0, 2.0, -10.0])

        result = logaddexp_onnx(x, y)
        expected = torch.logaddexp(x, y)

        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_large_difference(self):
        """Test numerical stability with large differences between x and y."""
        x = torch.tensor([100.0, -100.0, 50.0])
        y = torch.tensor([0.0, 0.0, -50.0])

        result = logaddexp_onnx(x, y)
        expected = torch.logaddexp(x, y)

        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_negative_values(self):
        """Test with negative values."""
        x = torch.tensor([-1.0, -5.0, -10.0])
        y = torch.tensor([-2.0, -3.0, -8.0])

        result = logaddexp_onnx(x, y)
        expected = torch.logaddexp(x, y)

        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_zero_values(self):
        """Test with zero values."""
        x = torch.tensor([0.0, 0.0])
        y = torch.tensor([0.0, 1.0])

        result = logaddexp_onnx(x, y)
        expected = torch.logaddexp(x, y)

        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_2d_tensor(self):
        """Test with 2D tensors."""
        x = torch.randn(4, 8)
        y = torch.randn(4, 8)

        result = logaddexp_onnx(x, y)
        expected = torch.logaddexp(x, y)

        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_3d_tensor(self):
        """Test with 3D tensors (typical for audio models)."""
        x = torch.randn(2, 100, 100)  # (batch, time, features)
        y = torch.randn(2, 100, 100)

        result = logaddexp_onnx(x, y)
        expected = torch.logaddexp(x, y)

        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_broadcasting(self):
        """Test broadcasting behavior."""
        x = torch.randn(4, 1)
        y = torch.randn(1, 8)

        result = logaddexp_onnx(x, y)
        expected = torch.logaddexp(x, y)

        assert result.shape == (4, 8)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_single_element(self):
        """Test with single element tensors."""
        x = torch.tensor(1.5)
        y = torch.tensor(2.5)

        result = logaddexp_onnx(x, y)
        expected = torch.logaddexp(x, y)

        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

    def test_no_nan_or_inf(self):
        """Test that results don't contain NaN or Inf for reasonable inputs."""
        x = torch.randn(100, 100) * 10  # Values in range [-30, 30] roughly
        y = torch.randn(100, 100) * 10

        result = logaddexp_onnx(x, y)

        assert not torch.isnan(result).any(), "Result contains NaN"
        assert not torch.isinf(result).any(), "Result contains Inf"


class TestLogaddexpOnnxGradient:
    """Test gradient computation for logaddexp_onnx."""

    def test_gradient_computation(self):
        """Test that gradients can be computed through logaddexp_onnx."""
        x = torch.randn(4, 8, requires_grad=True)
        y = torch.randn(4, 8, requires_grad=True)

        result = logaddexp_onnx(x, y)
        loss = result.sum()
        loss.backward()

        assert x.grad is not None, "Gradient for x was not computed"
        assert y.grad is not None, "Gradient for y was not computed"
        assert not torch.isnan(x.grad).any(), "Gradient for x contains NaN"
        assert not torch.isnan(y.grad).any(), "Gradient for y contains NaN"
