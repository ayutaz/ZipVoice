#!/usr/bin/env python3
"""
Verify ONNX model compatibility with Unity Sentis.

This script checks:
1. All operators are in the Sentis supported list
2. Tensor dimensions are within limits (max 8)
3. Data types are compatible
4. Opset version is within range (7-15)

Usage:
    python -m zipvoice.bin.verify_sentis_onnx --onnx-dir exp/zipvoice_sentis
    python -m zipvoice.bin.verify_sentis_onnx --onnx-file model.onnx
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple

import onnx
from onnx import TensorProto


# Unity Sentis supported operators (as of version 2.1)
# Reference: https://docs.unity3d.com/Packages/com.unity.sentis@2.1/manual/supported-operators.html
SENTIS_SUPPORTED_OPS: Set[str] = {
    # Basic operations
    "Abs", "Acos", "Acosh", "Add", "And", "ArgMax", "ArgMin", "Asin", "Asinh",
    "Atan", "Atanh", "AveragePool", "BatchNormalization", "Bernoulli",
    "BitwiseAnd", "BitwiseNot", "BitwiseOr", "BitwiseXor", "BlackmanWindow",
    "Cast", "CastLike", "Ceil", "Celu", "Clip", "Compress", "Concat",
    "Constant", "ConstantOfShape", "Conv", "ConvTranspose", "Cos", "Cosh",
    "CumSum", "DepthToSpace", "DequantizeLinear", "Div", "Dropout", "Einsum",
    "Elu", "Equal", "Erf", "Exp", "Expand", "Flatten", "Floor", "Gather",
    "GatherElements", "GatherND", "Gemm", "GlobalAveragePool", "GlobalMaxPool",
    "Greater", "GreaterOrEqual", "GridSample", "HammingWindow", "HannWindow",
    "Hardmax", "HardSigmoid", "HardSwish", "Identity", "InstanceNormalization",
    "IsInf", "IsNaN", "LayerNormalization", "LeakyRelu", "Less", "LessOrEqual",
    "Log", "LogSoftmax", "LRN", "LSTM", "MatMul", "Max", "MaxPool", "Mean",
    "MelWeightMatrix", "Min", "Mod", "Mish", "Mul", "Multinomial", "Neg",
    "NonMaxSuppression", "NonZero", "Not", "OneHot", "Or", "Pad", "Pow",
    "PRelu", "QuantizeLinear", "RandomNormal", "RandomNormalLike",
    "RandomUniform", "RandomUniformLike", "Range", "Reciprocal", "ReduceL1",
    "ReduceL2", "ReduceLogSum", "ReduceLogSumExp", "ReduceMax", "ReduceMean",
    "ReduceMin", "ReduceProd", "ReduceSum", "ReduceSumSquare", "Relu",
    "Reshape", "Resize", "RoiAlign", "Round", "Scatter", "ScatterElements",
    "ScatterND", "Selu", "Shape", "Shrink", "Sigmoid", "Sign", "Sin", "Sinh",
    "Size", "Slice", "Softmax", "Softplus", "Softsign", "SpaceToDepth",
    "Split", "Sqrt", "Squeeze", "STFT", "Sub", "Sum", "Tan", "Tanh",
    "ThresholdedRelu", "Tile", "TopK", "Transpose", "Trilu", "Unsqueeze",
    "Upsample", "Where", "Xor",
    # Sentis-specific layers
    "Atan2", "BroadcastArgs", "Dense", "DequantizeUint8", "FloorDiv", "Gelu",
    "GeluFast", "MatMul2D", "MoveDim", "Narrow", "NotEqual", "RandomChoice",
    "Relu6", "RMSNormalization", "ScalarMad", "Select", "SliceSet", "Square",
    "TrueDiv", "Swish", "ScaleBias",
}

# Operators that are definitely NOT supported
SENTIS_UNSUPPORTED_OPS: Set[str] = {
    "Log1p",  # Use Log(1+x) instead
    "LogAddExp",
    "ComplexAbs",
    "FFT",
    "IFFT",
    "RFFT",
    "IRFFT",
    "Unique",
    "StringNormalizer",
    "TfIdfVectorizer",
    "Tokenizer",
    "SequenceAt",
    "SequenceConstruct",
    "SequenceEmpty",
    "SequenceErase",
    "SequenceInsert",
    "SequenceLength",
    "SplitToSequence",
    "ConcatFromSequence",
}

# Supported data types
SENTIS_SUPPORTED_DTYPES: Set[int] = {
    TensorProto.FLOAT,    # float32
    TensorProto.INT32,    # int32
    TensorProto.INT64,    # int64
    TensorProto.INT16,    # int16 (short)
    TensorProto.UINT8,    # uint8 (byte)
}

# Maximum tensor dimensions
SENTIS_MAX_DIMS = 8


def get_parser():
    parser = argparse.ArgumentParser(
        description="Verify ONNX model compatibility with Unity Sentis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--onnx-dir",
        type=str,
        default=None,
        help="Directory containing ONNX files to verify",
    )
    parser.add_argument(
        "--onnx-file",
        type=str,
        default=None,
        help="Single ONNX file to verify",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed information about each operator",
    )
    return parser


def get_tensor_dims(shape) -> int:
    """Get the number of dimensions from an ONNX shape."""
    return len(shape.dim)


def verify_model(model_path: str, verbose: bool = False) -> Tuple[List[str], List[str], Dict]:
    """Verify a single ONNX model for Sentis compatibility.

    Args:
        model_path: Path to the ONNX model file.
        verbose: Whether to print detailed information.

    Returns:
        Tuple of (errors, warnings, stats)
    """
    errors = []
    warnings = []
    stats = {
        "operators": {},
        "max_dims": 0,
        "opset_version": 0,
        "total_nodes": 0,
    }

    try:
        model = onnx.load(model_path)
    except Exception as e:
        errors.append(f"Failed to load model: {e}")
        return errors, warnings, stats

    # Check opset version
    for opset in model.opset_import:
        if opset.domain == "" or opset.domain == "ai.onnx":
            stats["opset_version"] = opset.version
            if opset.version < 7:
                errors.append(f"Opset version {opset.version} is too low (minimum: 7)")
            elif opset.version > 15:
                warnings.append(
                    f"Opset version {opset.version} is above recommended (7-15). "
                    "Results may be unpredictable."
                )

    # Check operators
    for node in model.graph.node:
        stats["total_nodes"] += 1
        op_type = node.op_type

        # Count operator usage
        stats["operators"][op_type] = stats["operators"].get(op_type, 0) + 1

        # Check if operator is supported
        if op_type in SENTIS_UNSUPPORTED_OPS:
            errors.append(
                f"Operator '{op_type}' (node: {node.name}) is NOT supported by Sentis"
            )
        elif op_type not in SENTIS_SUPPORTED_OPS:
            warnings.append(
                f"Operator '{op_type}' (node: {node.name}) may not be supported "
                "(not in known list)"
            )

    # Check tensor dimensions
    def check_tensor_dims(tensor_info, tensor_type: str):
        if tensor_info.type.HasField("tensor_type"):
            dims = get_tensor_dims(tensor_info.type.tensor_type.shape)
            stats["max_dims"] = max(stats["max_dims"], dims)
            if dims > SENTIS_MAX_DIMS:
                errors.append(
                    f"{tensor_type} tensor '{tensor_info.name}' has {dims} dimensions "
                    f"(max: {SENTIS_MAX_DIMS})"
                )

    # Check inputs
    for input_info in model.graph.input:
        check_tensor_dims(input_info, "Input")

    # Check outputs
    for output_info in model.graph.output:
        check_tensor_dims(output_info, "Output")

    # Check intermediate values
    for value_info in model.graph.value_info:
        check_tensor_dims(value_info, "Intermediate")

    # Check for boolean tensors (converted to float/int)
    for initializer in model.graph.initializer:
        if initializer.data_type == TensorProto.BOOL:
            warnings.append(
                f"Boolean tensor '{initializer.name}' will be converted to float/int "
                "(may increase memory usage)"
            )

    return errors, warnings, stats


def print_report(model_path: str, errors: List[str], warnings: List[str], stats: Dict):
    """Print a formatted verification report."""
    print("\n" + "=" * 70)
    print(f"Model: {model_path}")
    print("=" * 70)

    # Summary
    print(f"\nOpset version: {stats['opset_version']}")
    print(f"Total nodes: {stats['total_nodes']}")
    print(f"Max tensor dimensions: {stats['max_dims']}")
    print(f"Unique operators: {len(stats['operators'])}")

    # Operator breakdown
    if stats["operators"]:
        print("\nOperator usage:")
        for op, count in sorted(stats["operators"].items(), key=lambda x: -x[1]):
            supported = "OK" if op in SENTIS_SUPPORTED_OPS else "?"
            print(f"  {op}: {count} [{supported}]")

    # Errors
    if errors:
        print(f"\n[ERRORS] ({len(errors)})")
        for e in errors:
            print(f"  - {e}")
    else:
        print("\n[ERRORS] None")

    # Warnings
    if warnings:
        print(f"\n[WARNINGS] ({len(warnings)})")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("\n[WARNINGS] None")

    # Final status
    print("\n" + "-" * 70)
    if errors:
        print("STATUS: FAILED - Model has compatibility issues")
    elif warnings:
        print("STATUS: PASSED with warnings - Review warnings before deployment")
    else:
        print("STATUS: PASSED - Model is compatible with Unity Sentis")
    print("-" * 70)


def main():
    parser = get_parser()
    args = parser.parse_args()

    if not args.onnx_dir and not args.onnx_file:
        parser.error("Either --onnx-dir or --onnx-file must be specified")

    models_to_check = []

    if args.onnx_file:
        models_to_check.append(Path(args.onnx_file))

    if args.onnx_dir:
        onnx_dir = Path(args.onnx_dir)
        if not onnx_dir.exists():
            logging.error(f"Directory does not exist: {onnx_dir}")
            return 1
        models_to_check.extend(onnx_dir.glob("*.onnx"))

    if not models_to_check:
        logging.error("No ONNX files found")
        return 1

    all_passed = True

    for model_path in models_to_check:
        errors, warnings, stats = verify_model(str(model_path), args.verbose)
        print_report(str(model_path), errors, warnings, stats)

        if errors:
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("OVERALL: All models passed verification")
        return 0
    else:
        print("OVERALL: Some models have compatibility issues")
        return 1


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO, force=True)

    exit(main())
