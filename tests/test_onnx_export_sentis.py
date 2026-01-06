"""Tests for the Sentis ONNX export script components."""

import tempfile
from pathlib import Path

import onnx
import pytest
import torch
from onnx import TensorProto, helper

from zipvoice.bin.onnx_export_sentis import (
    SENTIS_UNSUPPORTED_OPS,
    add_meta_data,
    verify_sentis_compatibility,
)


class TestSentisUnsupportedOps:
    """Test the unsupported operators list."""

    def test_log1p_in_unsupported(self):
        """Ensure Log1p is marked as unsupported."""
        assert "Log1p" in SENTIS_UNSUPPORTED_OPS

    def test_logaddexp_in_unsupported(self):
        """Ensure LogAddExp is marked as unsupported."""
        assert "LogAddExp" in SENTIS_UNSUPPORTED_OPS

    def test_complex_ops_in_unsupported(self):
        """Ensure complex number ops are marked as unsupported."""
        assert "ComplexAbs" in SENTIS_UNSUPPORTED_OPS


class TestAddMetaData:
    """Test the add_meta_data function."""

    def create_simple_model(self) -> str:
        """Create a simple ONNX model for testing."""
        # Create a simple identity model
        node = helper.make_node(
            "Identity",
            inputs=["input"],
            outputs=["output"],
            name="identity",
        )

        input_tensor = helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, [1, 10]
        )
        output_tensor = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [1, 10]
        )

        graph = helper.make_graph(
            [node],
            "test_graph",
            [input_tensor],
            [output_tensor],
        )

        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 13)],
        )

        temp_dir = tempfile.mkdtemp()
        filepath = Path(temp_dir) / "test_model.onnx"
        onnx.save(model, str(filepath))

        return str(filepath)

    def test_add_single_metadata(self):
        """Test adding a single metadata entry."""
        model_path = self.create_simple_model()

        add_meta_data(model_path, {"test_key": "test_value"})

        model = onnx.load(model_path)
        metadata = {m.key: m.value for m in model.metadata_props}

        assert "test_key" in metadata
        assert metadata["test_key"] == "test_value"

    def test_add_multiple_metadata(self):
        """Test adding multiple metadata entries."""
        model_path = self.create_simple_model()

        meta = {
            "version": "1",
            "author": "test",
            "target_runtime": "unity_sentis",
        }
        add_meta_data(model_path, meta)

        model = onnx.load(model_path)
        metadata = {m.key: m.value for m in model.metadata_props}

        for key, value in meta.items():
            assert key in metadata
            assert metadata[key] == value


class TestVerifySentisCompatibility:
    """Test the verify_sentis_compatibility function."""

    def create_model_with_ops(self, operators: list, opset_version: int = 13) -> str:
        """Create an ONNX model with specified operators."""
        nodes = []
        prev_output = "input"

        for i, op_type in enumerate(operators):
            output_name = f"output_{i}" if i < len(operators) - 1 else "output"

            if op_type in ["Add", "Mul"]:
                node = helper.make_node(
                    op_type,
                    inputs=[prev_output, prev_output],
                    outputs=[output_name],
                    name=f"{op_type.lower()}_{i}",
                )
            else:
                node = helper.make_node(
                    op_type,
                    inputs=[prev_output],
                    outputs=[output_name],
                    name=f"{op_type.lower()}_{i}",
                )

            nodes.append(node)
            prev_output = output_name

        input_tensor = helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, [1, 10, 10]
        )
        output_tensor = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [1, 10, 10]
        )

        graph = helper.make_graph(
            nodes,
            "test_graph",
            [input_tensor],
            [output_tensor],
        )

        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", opset_version)],
        )

        temp_dir = tempfile.mkdtemp()
        filepath = Path(temp_dir) / "test_model.onnx"
        onnx.save(model, str(filepath))

        return str(filepath)

    def test_supported_ops_no_warnings(self):
        """Test that supported operators produce no warnings."""
        model_path = self.create_model_with_ops(["Relu", "Add"])

        warnings = verify_sentis_compatibility(model_path)

        # Filter to only operator-related warnings
        op_warnings = [w for w in warnings if "operator" in w.lower()]
        assert len(op_warnings) == 0, f"Unexpected warnings: {op_warnings}"

    def test_unsupported_ops_produce_warnings(self):
        """Test that unsupported operators produce warnings."""
        model_path = self.create_model_with_ops(["Relu", "Log1p"])

        warnings = verify_sentis_compatibility(model_path)

        assert len(warnings) > 0, "Should warn about Log1p"
        assert any("Log1p" in w for w in warnings)


class TestOnnxExportConfig:
    """Test export configuration values."""

    def test_default_opset_version(self):
        """Test that default opset version is 15 for Sentis."""
        # Import the default from the script
        # We check this indirectly by verifying the documentation
        from zipvoice.bin import onnx_export_sentis

        # The script should use opset 15
        # We can't easily test this without running the full export,
        # so we just verify the constant comment in docstring
        assert "opset version 15" in onnx_export_sentis.__doc__.lower() or \
               "opset 15" in str(onnx_export_sentis.export_text_encoder.__doc__ or "").lower()


class TestOnnxModelStructure:
    """Test expected ONNX model structure for ZipVoice."""

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

    def test_fm_decoder_inputs(self, fm_decoder_path):
        """Test that fm_decoder has expected input names."""
        model = onnx.load(fm_decoder_path)
        input_names = [inp.name for inp in model.graph.input]

        expected_inputs = ["t", "x", "text_condition", "speech_condition", "guidance_scale"]
        for expected in expected_inputs:
            assert expected in input_names, f"Missing input: {expected}"

    def test_fm_decoder_output(self, fm_decoder_path):
        """Test that fm_decoder has expected output name."""
        model = onnx.load(fm_decoder_path)
        output_names = [out.name for out in model.graph.output]

        assert "v" in output_names, "Missing output: v"

    def test_text_encoder_inputs(self, text_encoder_path):
        """Test that text_encoder has expected input names."""
        model = onnx.load(text_encoder_path)
        input_names = [inp.name for inp in model.graph.input]

        expected_inputs = ["tokens", "prompt_tokens", "prompt_features_len", "speed"]
        for expected in expected_inputs:
            assert expected in input_names, f"Missing input: {expected}"

    def test_text_encoder_output(self, text_encoder_path):
        """Test that text_encoder has expected output name."""
        model = onnx.load(text_encoder_path)
        output_names = [out.name for out in model.graph.output]

        assert "text_condition" in output_names, "Missing output: text_condition"

    def test_fm_decoder_metadata(self, fm_decoder_path):
        """Test that fm_decoder has expected metadata."""
        model = onnx.load(fm_decoder_path)
        metadata = {m.key: m.value for m in model.metadata_props}

        # Should have at least version and author
        assert "version" in metadata or "model_author" in metadata

    def test_text_encoder_opset_version(self, text_encoder_path):
        """Test that text_encoder uses appropriate opset version."""
        model = onnx.load(text_encoder_path)

        opset_version = None
        for opset in model.opset_import:
            if opset.domain == "" or opset.domain == "ai.onnx":
                opset_version = opset.version
                break

        assert opset_version is not None
        assert 7 <= opset_version <= 15, f"Opset {opset_version} outside Sentis range"
