"""Tests for the Sentis ONNX verification script."""

import tempfile
from pathlib import Path

import onnx
import pytest
from onnx import TensorProto, helper

from zipvoice.bin.verify_sentis_onnx import (
    SENTIS_LIMITED_SUPPORT_OPS,
    SENTIS_SUPPORTED_OPS,
    SENTIS_UNSUPPORTED_OPS,
    verify_model,
)


def create_simple_onnx_model(
    operators: list,
    opset_version: int = 13,
    filename: str = "test_model.onnx",
) -> str:
    """Create a simple ONNX model with specified operators for testing.

    Args:
        operators: List of (op_type, node_name) tuples
        opset_version: ONNX opset version
        filename: Output filename

    Returns:
        Path to the created ONNX model
    """
    # Create a simple graph with the specified operators
    nodes = []
    prev_output = "input"

    for i, (op_type, node_name) in enumerate(operators):
        output_name = f"output_{i}" if i < len(operators) - 1 else "output"

        if op_type == "Add":
            node = helper.make_node(
                op_type,
                inputs=[prev_output, prev_output],
                outputs=[output_name],
                name=node_name,
            )
        elif op_type == "MatMul":
            node = helper.make_node(
                op_type,
                inputs=[prev_output, prev_output],
                outputs=[output_name],
                name=node_name,
            )
        elif op_type == "Log1p":
            # Unsupported operator
            node = helper.make_node(
                op_type,
                inputs=[prev_output],
                outputs=[output_name],
                name=node_name,
            )
        elif op_type == "MatMulInteger":
            # Limited support operator
            node = helper.make_node(
                op_type,
                inputs=[prev_output, prev_output],
                outputs=[output_name],
                name=node_name,
            )
        else:
            # Generic unary operator
            node = helper.make_node(
                op_type,
                inputs=[prev_output],
                outputs=[output_name],
                name=node_name,
            )

        nodes.append(node)
        prev_output = output_name

    # Create input/output
    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 10, 10]
    )
    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 10, 10]
    )

    # Create graph
    graph = helper.make_graph(
        nodes,
        "test_graph",
        [input_tensor],
        [output_tensor],
    )

    # Create model
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", opset_version)],
    )

    # Save to temp file
    temp_dir = tempfile.mkdtemp()
    filepath = Path(temp_dir) / filename
    onnx.save(model, str(filepath))

    return str(filepath)


class TestSentisOperatorLists:
    """Test that operator lists are properly defined."""

    def test_supported_ops_not_empty(self):
        """Ensure supported operators list is not empty."""
        assert len(SENTIS_SUPPORTED_OPS) > 0

    def test_unsupported_ops_not_empty(self):
        """Ensure unsupported operators list is not empty."""
        assert len(SENTIS_UNSUPPORTED_OPS) > 0

    def test_limited_support_ops_defined(self):
        """Ensure limited support operators list exists."""
        assert len(SENTIS_LIMITED_SUPPORT_OPS) > 0

    def test_no_overlap_supported_unsupported(self):
        """Ensure no overlap between supported and unsupported lists."""
        overlap = SENTIS_SUPPORTED_OPS & SENTIS_UNSUPPORTED_OPS
        assert len(overlap) == 0, f"Overlap found: {overlap}"

    def test_common_ops_in_supported(self):
        """Ensure common operators are in supported list."""
        common_ops = {"Add", "MatMul", "Relu", "Softmax", "Conv", "Reshape"}
        for op in common_ops:
            assert op in SENTIS_SUPPORTED_OPS, f"{op} should be in supported list"

    def test_log1p_in_unsupported(self):
        """Ensure Log1p is in unsupported list (key for our fix)."""
        assert "Log1p" in SENTIS_UNSUPPORTED_OPS


class TestVerifyModel:
    """Test the verify_model function."""

    def test_model_with_supported_ops(self):
        """Test verification of model with only supported operators."""
        model_path = create_simple_onnx_model(
            operators=[
                ("Relu", "relu_1"),
                ("Add", "add_1"),
            ],
            opset_version=13,
        )

        errors, warnings, stats = verify_model(model_path)

        assert len(errors) == 0, f"Unexpected errors: {errors}"
        assert stats["opset_version"] == 13
        assert stats["total_nodes"] == 2

    def test_model_with_unsupported_ops(self):
        """Test that unsupported operators are detected as errors."""
        model_path = create_simple_onnx_model(
            operators=[
                ("Relu", "relu_1"),
                ("Log1p", "log1p_unsupported"),  # Unsupported
            ],
            opset_version=13,
        )

        errors, warnings, stats = verify_model(model_path)

        assert len(errors) > 0, "Should detect unsupported operator"
        assert any("Log1p" in e for e in errors)

    def test_opset_version_detection(self):
        """Test that opset version is correctly detected."""
        for version in [7, 13, 15]:
            model_path = create_simple_onnx_model(
                operators=[("Relu", "relu_1")],
                opset_version=version,
            )

            errors, warnings, stats = verify_model(model_path)

            assert stats["opset_version"] == version

    def test_opset_version_warning_above_15(self):
        """Test that opset version > 15 generates warning."""
        model_path = create_simple_onnx_model(
            operators=[("Relu", "relu_1")],
            opset_version=17,
        )

        errors, warnings, stats = verify_model(model_path)

        # Should have warning about opset version
        assert any("opset" in w.lower() for w in warnings)

    def test_opset_version_error_below_7(self):
        """Test that opset version < 7 generates error."""
        model_path = create_simple_onnx_model(
            operators=[("Relu", "relu_1")],
            opset_version=6,
        )

        errors, warnings, stats = verify_model(model_path)

        # Should have error about opset version
        assert any("opset" in e.lower() for e in errors)

    def test_operator_counting(self):
        """Test that operators are correctly counted."""
        model_path = create_simple_onnx_model(
            operators=[
                ("Relu", "relu_1"),
                ("Relu", "relu_2"),
                ("Add", "add_1"),
            ],
            opset_version=13,
        )

        errors, warnings, stats = verify_model(model_path)

        assert stats["operators"].get("Relu", 0) == 2
        assert stats["operators"].get("Add", 0) == 1

    def test_nonexistent_file(self):
        """Test handling of nonexistent file."""
        errors, warnings, stats = verify_model("/nonexistent/path/model.onnx")

        assert len(errors) > 0, "Should report error for nonexistent file"


class TestExistingOnnxModels:
    """Test verification of existing ONNX models in the project."""

    @pytest.fixture
    def fm_decoder_path(self):
        """Path to the fm_decoder ONNX model."""
        path = Path("exp/zipvoice_moe_90h_onnx/fm_decoder.onnx")
        if not path.exists():
            pytest.skip("fm_decoder.onnx not found")
        return str(path)

    @pytest.fixture
    def text_encoder_path(self):
        """Path to the text_encoder ONNX model."""
        path = Path("exp/zipvoice_moe_90h_onnx/text_encoder.onnx")
        if not path.exists():
            pytest.skip("text_encoder.onnx not found")
        return str(path)

    def test_fm_decoder_compatibility(self, fm_decoder_path):
        """Test that fm_decoder.onnx passes Sentis compatibility check."""
        errors, warnings, stats = verify_model(fm_decoder_path)

        assert len(errors) == 0, f"fm_decoder has errors: {errors}"
        assert stats["max_dims"] <= 8, "Tensor dimensions exceed Sentis limit"

    def test_text_encoder_compatibility(self, text_encoder_path):
        """Test that text_encoder.onnx passes Sentis compatibility check."""
        errors, warnings, stats = verify_model(text_encoder_path)

        assert len(errors) == 0, f"text_encoder has errors: {errors}"
        assert stats["max_dims"] <= 8, "Tensor dimensions exceed Sentis limit"

    def test_fm_decoder_no_log1p(self, fm_decoder_path):
        """Verify fm_decoder doesn't use Log1p operator."""
        errors, warnings, stats = verify_model(fm_decoder_path)

        assert "Log1p" not in stats["operators"], "Model should not use Log1p"

    def test_text_encoder_no_log1p(self, text_encoder_path):
        """Verify text_encoder doesn't use Log1p operator."""
        errors, warnings, stats = verify_model(text_encoder_path)

        assert "Log1p" not in stats["operators"], "Model should not use Log1p"
